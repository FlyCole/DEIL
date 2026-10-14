"""Locate externally fetched, pinned depth libraries without depending on cwd."""

from functools import lru_cache
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / 'depth_dependencies.json'


@lru_cache(maxsize=1)
def dependency_manifest():
    with MANIFEST_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


def dependency_path(name, required=True):
    manifest = dependency_manifest()
    canonical = manifest['variant_aliases'].get(name, name)
    if canonical not in manifest['dependencies']:
        raise ValueError(f'No external depth dependency is registered for {name!r}')
    entry = manifest['dependencies'][canonical]
    configured_root = os.environ.get('HAKI_DEPTH_DEPENDENCY_ROOT')
    source_root = (
        Path(configured_root).expanduser().resolve() if configured_root
        else PROJECT_ROOT / manifest['default_source_root']
    )
    candidates = [source_root / entry['directory']]
    # Existing ignored checkouts remain usable on the original machine. An
    # explicit root selects only that root, so missing files cannot silently
    # switch to a different checkout.
    if not configured_root:
        candidates.extend(PROJECT_ROOT / directory for directory in entry['legacy_directories'])
    for candidate in candidates:
        if (candidate / entry['import_marker']).is_file():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(
            f'Missing {canonical} source checkout. Run '
            f'python scripts/bootstrap_depth_dependencies.py --only {canonical}; '
            'set HAKI_DEPTH_DEPENDENCY_ROOT when using --root.'
        )
    return None


def activate_depth_dependency(name, required=False):
    """Add a source package root, preserving installed-package fallback."""
    path = dependency_path(
        name, required=required or bool(os.environ.get('HAKI_DEPTH_DEPENDENCY_ROOT'))
    )
    if path is not None:
        location = str(path)
        if location not in sys.path:
            sys.path.insert(0, location)
    return path


def activate_croco(required=False):
    """Pi3's optional ``models.curope`` lives in MASt3R's CroCo submodule."""
    mast3r = dependency_path('mast3r', required=required)
    if mast3r is None:
        return None
    croco = mast3r / 'dust3r' / 'croco'
    if not (croco / 'models' / 'curope' / '__init__.py').is_file():
        if required:
            raise FileNotFoundError('MASt3R CroCo source is missing; initialise recursive submodules.')
        return None
    if str(croco) not in sys.path:
        sys.path.insert(0, str(croco))
    return croco
