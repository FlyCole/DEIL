"""Crash-resumable publication of local Zarr 2 derived point-cloud arrays.

Supports DirectoryStore and NestedDirectoryStore on one local filesystem.
Writers must use this coordinator's advisory lock; source arrays must remain
immutable. Chunk writes and journal replacements protect against processing
failures and process termination, not power loss or concurrent reader snapshots
across several array renames. Readers should wait for publication to finish.
"""

from copy import deepcopy
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import numpy as np
import zarr


TRANSACTION_SCHEMA = 1
WORKSPACE = '.haki_derived_pcd'
JOURNAL_NAME = 'active.json'
_TRANSACTION_ID = re.compile(r'^[0-9a-f]{32}$')


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _same_config(previous, expected, key):
    if previous != expected:
        raise RuntimeError(
            f'Cannot safely resume /data/{key}: processing configuration or '
            'source fingerprint differs. Use --rebuild (without --resume) '
            'to construct a replacement while preserving the current output.'
        )


class _PublicationCoordinator:
    """Model-free store, journal and directory-publication coordination."""

    def __init__(self, root):
        if not isinstance(root, zarr.Group):
            raise ValueError('Use a root Zarr group for transactional preprocessing.')
        if not isinstance(root.store, zarr.storage.DirectoryStore):
            raise ValueError(
                'Transactional preprocessing supports only local Zarr 2 '
                'DirectoryStore/NestedDirectoryStore; ZipStore, MemoryStore '
                'and remote stores require export to a local directory first.'
            )
        if root.path or root.chunk_store is not root.store:
            raise ValueError('Use the root group with metadata and chunks in the same local store.')
        self.root = root
        self.store = root.store
        self.root_path = Path(self.store.path).resolve()
        self.workspace = self.root_path / WORKSPACE
        self.journal_path = self.workspace / JOURNAL_NAME
        self.arrays = {}
        self.state = None
        self.lock = None

    def _acquire_writer_lock(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.lock = (self.workspace / '.lock').open('a+b')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError('Another derived point-cloud writer is active for this dataset.')

    def _read_state(self):
        if not self.journal_path.exists():
            return None
        with self.journal_path.open(encoding='utf-8') as handle:
            state = json.load(handle)
        if (not isinstance(state, dict)
                or state.get('schema_version') != TRANSACTION_SCHEMA
                or not _TRANSACTION_ID.fullmatch(str(state.get('transaction_id', '')))
                or state.get('phase') not in {'processing', 'prepared', 'publishing', 'complete'}
                or not isinstance(state.get('configs'), dict)
                or not state['configs']
                or any(not isinstance(config, dict) for config in state['configs'].values())
                or not isinstance(state.get('committed_steps'), dict)
                or isinstance(state.get('target_steps'), bool)
                or not isinstance(state.get('target_steps'), int)
                or state['target_steps'] <= 0):
            raise RuntimeError('Invalid derived point-cloud transaction journal; preserve it for inspection.')
        if set(state['configs']) != set(state['committed_steps']):
            raise RuntimeError('Invalid derived point-cloud transaction progress keys.')
        if any(not isinstance(key, str) or not re.fullmatch(r'point_cloud_[a-z0-9_]+', key)
               for key in state['configs']):
            raise RuntimeError('Invalid derived point-cloud transaction output key.')
        for count in state['committed_steps'].values():
            if (isinstance(count, bool) or not isinstance(count, int)
                    or not 0 <= count <= state['target_steps']):
                raise RuntimeError('Invalid persistent committed_steps in transaction journal.')
            if state['phase'] != 'processing' and count != state['target_steps']:
                raise RuntimeError('Publication journal contains an incomplete output population.')
        return state

    def _stage_key(self, state, key):
        return f'{WORKSPACE}/{state["transaction_id"]}/staging/{key}'

    def _backup_path(self, state, key):
        return self.workspace / state['transaction_id'] / 'backup' / key

    def _write_state(self, state):
        _atomic_json(self.journal_path, state)
        self.state = state

    def _recover_publication(self, state):
        """Complete directory renames; originals survive until all installs finish."""
        self._preflight_publication(state)
        for key in state['configs']:
            staged = self.root_path / self._stage_key(state, key)
            prepared_path = staged if staged.exists() else self.root_path / 'data' / key
            self._persist_chunk_separator(self._require_array_directory(prepared_path, key))
        state = deepcopy(state)
        state['phase'] = 'publishing'
        self._write_state(state)
        for key in state['configs']:
            staged = self.root_path / self._stage_key(state, key)
            target = self.root_path / 'data' / key
            backup = self._backup_path(state, key)
            if staged.exists():
                prepared = zarr.open_array(store=self.store, path=self._stage_key(state, key), mode='r')
                self._validate_prepared_array(prepared, state, key)
                if target.exists():
                    if backup.exists():
                        raise RuntimeError(f'Ambiguous interrupted publication for {key}; preserve transaction files.')
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backup)
                os.replace(staged, target)
            if not target.is_dir():
                raise RuntimeError(f'Missing published target during recovery: {key}')
            installed = zarr.open_array(store=self.store, path=f'data/{key}', mode='r')
            self._validate_prepared_array(installed, state, key)
        state['phase'] = 'complete'
        self._write_state(state)
        shutil.rmtree(self.workspace / state['transaction_id'], ignore_errors=True)
        try:
            zarr.consolidate_metadata(self.store)
        except Exception:
            # Old consolidated shapes must never survive a successful replacement.
            if '.zmetadata' in self.store:
                del self.store['.zmetadata']
        self.journal_path.unlink(missing_ok=True)

    def _validate_prepared_array(self, array, state, key):
        config = state['configs'][key]
        points = config.get('num_points', array.shape[1] if array.ndim == 3 else None)
        if (array.attrs.get('derived_transaction_id') != state['transaction_id']
                or array.attrs.get('derived_transaction_schema') != TRANSACTION_SCHEMA
                or array.attrs.get('derived_processing_config') != config
                or array.attrs.get('committed_steps') != state['committed_steps'][key]
                or isinstance(points, bool) or not isinstance(points, int) or points <= 0
                or array.shape != (state['committed_steps'][key], points, 3)
                or array.dtype != np.dtype('float32')):
            raise RuntimeError(f'Invalid prepared/published metadata or population for {key}; backups are preserved.')

    def _require_array_directory(self, path, key):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f'Invalid publication directory for {key}: {path}; preserve transaction files.')
        try:
            return zarr.open_array(store=self.store,
                                   path=path.relative_to(self.root_path).as_posix(), mode='r')
        except Exception as error:
            raise RuntimeError(f'Invalid publication array for {key}: {path}; preserve transaction files.') from error

    def _persist_chunk_separator(self, array):
        # Older zarr.open_array calls omit this field. Ordinary NestedStore
        # reads infer '/', while consolidated readers then assume '.' and
        # silently return fill values. Record the verified array's separator.
        metadata_key = array.path + '/.zarray'
        metadata = json.loads(self.store[metadata_key])
        if 'dimension_separator' not in metadata:
            metadata['dimension_separator'] = array._dimension_separator
            self.store[metadata_key] = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()

    def _validate_committed_chunks(self, array, key):
        """Reject missing or unreadable staged chunks before moving any original."""
        chunk_counts = [range((size + chunk - 1) // chunk)
                        for size, chunk in zip(array.shape, array.chunks)]
        for coordinates in product(*chunk_counts):
            if array._chunk_key(coordinates) not in array.chunk_store:
                raise RuntimeError(f'Missing committed point-cloud chunk for {key}; backups are preserved.')
        try:
            for start in range(0, array.shape[0], array.chunks[0]):
                if not np.isfinite(array[start:min(start + array.chunks[0], array.shape[0])]).all():
                    raise RuntimeError(f'Nonfinite committed point-cloud values for {key}.')
        except Exception as error:
            raise RuntimeError(f'Unreadable/invalid committed point-cloud chunks for {key}; backups are preserved.') from error

    def _preflight_publication(self, state, verify_chunks=False):
        """Validate every output before publishing the first one."""
        for key in state['configs']:
            staged = self.root_path / self._stage_key(state, key)
            target = self.root_path / 'data' / key
            backup = self._backup_path(state, key)
            for path in (staged, target, backup):
                relative = path.relative_to(self.root_path)
                for index in range(1, len(relative.parts) + 1):
                    if self.root_path.joinpath(*relative.parts[:index]).is_symlink():
                        raise RuntimeError(f'Symlink in publication path for {key}; preserve transaction files.')
            if staged.exists():
                prepared = self._require_array_directory(staged, key)
                self._validate_prepared_array(prepared, state, key)
                if target.exists():
                    if backup.exists():
                        raise RuntimeError(f'Ambiguous interrupted publication for {key}; preserve transaction files.')
                    original = self._require_array_directory(target, key)
                    if original.attrs.get('derived_transaction_id') == state['transaction_id']:
                        raise RuntimeError(f'Duplicate prepared/published output for {key}; preserve transaction files.')
            elif target.exists():
                prepared = self._require_array_directory(target, key)
                self._validate_prepared_array(prepared, state, key)
            else:
                raise RuntimeError(f'Missing prepared/published output for {key}; backups are preserved.')
            if backup.exists():
                original = self._require_array_directory(backup, key)
                if original.attrs.get('derived_transaction_id') == state['transaction_id']:
                    raise RuntimeError(f'Unexpected replacement in backup for {key}; preserve transaction files.')
            if verify_chunks:
                self._validate_committed_chunks(prepared, key)

    def close(self):
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None



class DerivedPCDTransaction(_PublicationCoordinator):
    """Coordinate one multi-estimator preprocessing transaction per store.

    Construction and ``preflight`` are read-only. Call ``begin`` only after
    all parameters, input observations, and estimator initialization succeed.
    ``append`` advances progress only after a successful row write. A failed
    job keeps its hidden staging arrays for ``--resume`` and leaves published
    arrays untouched. ``publish`` runs only after every requested row exists.
    """

    def __init__(self, root, configs, num_points, target_steps, resume=False):
        super().__init__(root)
        if (not configs or any(not re.fullmatch(r'point_cloud_[a-z0-9_]+', str(key))
                               for key in configs)):
            raise ValueError('Expected nonempty derived point_cloud_<estimator> output configurations.')
        if int(num_points) <= 0 or int(target_steps) <= 0:
            raise ValueError('num_points and target_steps must be positive.')
        json.dumps(configs, allow_nan=False)
        self.configs = deepcopy(configs)
        self.num_points = int(num_points)
        self.target_steps = int(target_steps)
        self.resume = bool(resume)




    def _published_progress(self):
        progress = {}
        data = self.root['data']
        for key, config in self.configs.items():
            if not self.resume or key not in data:
                progress[key] = 0
                continue
            array = data[key]
            if (array.attrs.get('derived_transaction_schema') != TRANSACTION_SCHEMA
                    or 'committed_steps' not in array.attrs):
                raise RuntimeError(
                    f'Cannot safely resume legacy /data/{key}: it has no durable '
                    'commit marker, so its length cannot prove completed writes. '
                    'Use --rebuild (without --resume) for a protected replacement; '
                    'or export and independently verify legacy rows before migration.'
                )
            _same_config(array.attrs.get('derived_processing_config'), config, key)
            count = array.attrs['committed_steps']
            if (isinstance(count, bool) or not isinstance(count, int)
                    or count < 0 or count > array.shape[0]
                    or array.shape[1:] != (self.num_points, 3)
                    or array.dtype != np.dtype('float32')):
                raise RuntimeError(f'Invalid committed published array /data/{key}.')
            if count > self.target_steps:
                raise RuntimeError(f'/data/{key} already has {count} committed rows; --limit_steps cannot shorten a resume.')
            progress[key] = count
        return progress

    def preflight(self):
        """Reject incompatible resumes without creating or modifying any key."""
        state = self._read_state()
        if state is not None and self.resume:
            if state['configs'] != self.configs:
                raise RuntimeError(
                    'Pending derived transaction has different estimators, processing '
                    'settings, or source fingerprint. Resume its original configuration '
                    'or use --rebuild to restart safely.'
                )
            for key, count in state['committed_steps'].items():
                if count > self.target_steps:
                    raise RuntimeError(f'{key} has {count} committed rows; --limit_steps cannot shorten a resume.')
                stage_key = self._stage_key(state, key)
                if state['phase'] == 'processing':
                    if stage_key not in self.root and count:
                        raise RuntimeError(f'Missing staged array with committed rows: {stage_key}')
                    if stage_key in self.root:
                        array = self.root[stage_key]
                        if (array.shape[0] < count or array.shape[1:] != (self.num_points, 3)
                                or array.dtype != np.dtype('float32')):
                            raise RuntimeError(f'Invalid staged shape/progress: {stage_key}')
            return dict(state['committed_steps'])
        return self._published_progress()


    def _open_stage(self, key):
        path = self._stage_key(self.state, key)
        if path in self.root:
            return zarr.open_array(store=self.store, path=path, mode='a')
        return zarr.open_array(
            store=self.store, path=path, mode='w', shape=(0, self.num_points, 3),
            chunks=(100, self.num_points, 3), dtype='float32',
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1),
            dimension_separator=getattr(self.store, '_dimension_separator', None) or '.',
        )

    def begin(self):
        """Acquire the writer lock, revalidate, and allocate/resume staging."""
        self._acquire_writer_lock()
        try:
            self.preflight()
            pending = self._read_state()
            if pending is not None and pending['phase'] != 'processing':
                self._recover_publication(pending)
                pending = None
            if pending is not None and not self.resume:
                shutil.rmtree(self.workspace / pending['transaction_id'], ignore_errors=True)
                self.journal_path.unlink()
                pending = None
            if pending is None:
                progress = self._published_progress()
                self._write_state(dict(schema_version=TRANSACTION_SCHEMA,
                    transaction_id=uuid.uuid4().hex, phase='processing',
                    configs=self.configs, committed_steps={key: 0 for key in self.configs},
                    target_steps=self.target_steps))
                for key, count in progress.items():
                    array = self._open_stage(key)
                    self.arrays[key] = array
                    if count:
                        array.resize((count, self.num_points, 3))
                        original = self.root['data'][key]
                        for offset in range(0, count, 100):
                            array[offset:offset + 100] = original[offset:min(offset + 100, count)]
                        state = deepcopy(self.state)
                        state['committed_steps'][key] = count
                        self._write_state(state)
            else:
                self.state = pending
                for key, count in pending['committed_steps'].items():
                    array = self._open_stage(key)
                    # A resize or chunk write may have succeeded before the commit
                    # journal update. Never infer completion from that tail length.
                    if array.shape[0] != count:
                        array.resize((count, self.num_points, 3))
                    self.arrays[key] = array
                state = deepcopy(self.state)
                state['target_steps'] = self.target_steps
                self._write_state(state)
            return self
        except BaseException:
            self.close()
            raise

    def committed_steps(self, key):
        return self.state['committed_steps'][key]

    def append(self, key, index, row):
        if self.state is None or self.state['phase'] != 'processing':
            raise RuntimeError('No active processing transaction.')
        if index != self.committed_steps(key):
            raise RuntimeError(f'Nonsequential row for {key}: {index}, expected {self.committed_steps(key)}')
        row = np.asarray(row, dtype=np.float32)
        if row.shape != (self.num_points, 3) or not np.isfinite(row).all():
            raise ValueError(f'Invalid derived row for {key}: expected {self.num_points} finite XYZ points.')
        array = self.arrays[key]
        array.resize((index + 1, self.num_points, 3))
        array[index] = row
        state = deepcopy(self.state)
        state['committed_steps'][key] = index + 1
        self._write_state(state)

    def publish(self):
        if any(count != self.target_steps for count in self.state['committed_steps'].values()):
            raise RuntimeError('Cannot publish an incomplete derived transaction.')
        for key, array in self.arrays.items():
            array.attrs.update({
                **self.configs[key].get('noise_metadata', {}),
                'derived_transaction_schema': TRANSACTION_SCHEMA,
                'derived_processing_config': self.configs[key],
                'committed_steps': self.target_steps,
                'derived_transaction_id': self.state['transaction_id'],
            })
        state = deepcopy(self.state)
        state['phase'] = 'prepared'
        self._write_state(state)
        self._recover_publication(state)




    def __enter__(self):
        return self.begin()

    def __exit__(self, *exception):
        self.close()


def recover_pending_publication(root):
    """Finish interrupted publication without loading any depth estimator.

    Processing and prepared journals are left unchanged. A publishing/complete
    journal is recovered only under the writer lock after all available staged,
    installed and backup arrays pass validation. This validates chunk presence,
    decoding and finite XYZ values, rather than promising power-loss durability
    or cryptographic verification of old legacy backup contents.
    """
    coordinator = _PublicationCoordinator(root)
    pending = coordinator._read_state()

    def report(state, recovered=False):
        return {
            'recovered': recovered,
            'phase': state['phase'] if state is not None else None,
            'transaction_id': state['transaction_id'] if state is not None else None,
            'configs': deepcopy(state['configs']) if state is not None else None,
            'target_steps': state['target_steps'] if state is not None else None,
        }

    if pending is None or pending['phase'] in {'processing', 'prepared'}:
        return report(pending)
    coordinator._acquire_writer_lock()
    try:
        # Another writer may have finished between the first read and the lock.
        pending = coordinator._read_state()
        if pending is None or pending['phase'] in {'processing', 'prepared'}:
            return report(pending)
        coordinator._preflight_publication(pending, verify_chunks=True)
        provenance = report(pending, recovered=True)
        coordinator._recover_publication(pending)
        return provenance
    finally:
        coordinator.close()
