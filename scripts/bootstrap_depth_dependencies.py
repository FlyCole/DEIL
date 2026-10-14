#!/usr/bin/env python3
"""Fetch pinned depth sources and their small audited patches.

Default operation only fetches sources. --check never changes source or the
Python environment. --install is explicit, uses the active Python, and constrains
existing Torch/Torchvision versions so public Instant Policy's ABI is not silently
changed. Model weights and simulator dependencies are outside this script.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / 'depth_dependencies.json'


def run(command, cwd=None, capture=False):
    result = subprocess.run(command, cwd=cwd, check=True,
                            text=True, capture_output=capture)
    return result.stdout.strip() if capture else None


def git(path, *arguments, capture=True):
    return run(['git', '--no-optional-locks', '-C', str(path), *arguments], capture=capture)


def load_manifest():
    with MANIFEST_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_paths(entry):
    for patch in entry.get('patches', []):
        path = PROJECT_ROOT / patch['path']
        if sha256(path) != patch['sha256']:
            raise ValueError(f'Audited patch checksum does not match: {path}')
        yield patch, path


def _verify_tree(path, commit, patches=(), allow_unpatched=False):
    if Path(git(path, 'rev-parse', '--show-toplevel')).resolve() != path.resolve():
        raise ValueError(f'{path} is not its own Git checkout; refusing a bundled or partial directory')
    actual_commit = git(path, 'rev-parse', 'HEAD')
    if actual_commit != commit:
        raise ValueError(f'{path}: expected commit {commit}, found {actual_commit}; existing source is not reset')
    allowed = {}
    for patch in patches:
        allowed.update(patch['files'])
    changed = git(path, 'diff', '--name-only', '--ignore-submodules=untracked', 'HEAD').splitlines()
    for filename in changed:
        source = path / filename
        if filename not in allowed or not source.is_file() or sha256(source) != allowed[filename]:
            raise ValueError(f'Unrecorded source modification in {path}: {filename}; existing changes are preserved')
    if not allow_unpatched:
        for filename, expected in allowed.items():
            if not (path / filename).is_file() or sha256(path / filename) != expected:
                raise ValueError(f'Required audited patch is missing in {path}: {filename}')


def _verify_submodules(path, specifications):
    for submodule in specifications:
        child = path / submodule['path']
        _verify_tree(child, submodule['commit'])
        _verify_submodules(child, submodule.get('submodules', []))


def verify_checkout(path, entry, allow_unpatched=False):
    # Verify patch bytes as well as source results, even in read-only mode.
    tuple(patch_paths(entry))
    _verify_tree(path, entry['commit'], entry.get('patches', []), allow_unpatched)
    _verify_submodules(path, entry.get('submodules', []))
    license_name = 'LICENSE.txt' if entry['directory'] == 'vggt' else 'LICENSE'
    if not (path / license_name).is_file():
        raise ValueError(f'Upstream license is missing: {path / license_name}')


def fetch_checkout(path, entry):
    if path.exists():
        verify_checkout(path, entry, allow_unpatched=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--filter=blob:none', '--no-checkout', entry['repository'], str(path)])
        git(path, 'checkout', '--detach', entry['commit'], capture=False)
        if entry.get('submodules'):
            git(path, 'submodule', 'update', '--init', '--recursive', capture=False)
        verify_checkout(path, entry, allow_unpatched=True)
    for patch, patch_path in patch_paths(entry):
        if all((path / filename).is_file() and sha256(path / filename) == expected
               for filename, expected in patch['files'].items()):
            continue
        git(path, 'apply', '--check', str(patch_path), capture=False)
        git(path, 'apply', str(patch_path), capture=False)
    verify_checkout(path, entry)


def locate_existing(source_root, entry, allow_legacy):
    candidates = [source_root / entry['directory']]
    if allow_legacy:
        candidates.extend(PROJECT_ROOT / directory for directory in entry['legacy_directories'])
    for path in candidates:
        if (path / entry['import_marker']).is_file():
            return path.resolve()
    raise FileNotFoundError(f'Missing {entry["directory"]} source. Run this script without --check to fetch it.')


def install_checkout(path, entry):
    # The upstream Pi3/VGGT quick-start requirements specify mutually different
    # Torch versions. The project uses an existing chosen GPU environment; keep
    # its Torch stack fixed and let pip report incompatible transitive requests.
    versions = {}
    for package in ('torch', 'torchvision'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError('Install the chosen Torch/Torchvision GPU environment before --install') from exc
    try:
        versions['torchaudio'] = importlib.metadata.version('torchaudio')
    except importlib.metadata.PackageNotFoundError:
        pass
    with tempfile.TemporaryDirectory(prefix='haki-depth-pip-') as directory:
        constraints = Path(directory) / 'gpu-stack.txt'
        constraints.write_text('numpy<2\n' + ''.join(f'{name}=={version}\n' for name, version in versions.items()))
        for index, requirement_name in enumerate(entry['install']['requirements']):
            requirements = path / requirement_name
            lines = [line for line in requirements.read_text().splitlines()
                     if not re.match(r'^\s*(torch|torchvision)(?:\s|[<=>!~\[]|$)', line)]
            filtered = Path(directory) / f'requirements-{index}.txt'
            filtered.write_text('\n'.join(lines) + '\n')
            run([sys.executable, '-m', 'pip', 'install', '-c', str(constraints), '-r', str(filtered)], cwd=path)
        if entry['install']['editable']:
            run([sys.executable, '-m', 'pip', 'install', '-c', str(constraints), '-e', str(path)], cwd=path)


def build_curope(path, entry):
    extension = entry.get('cuda_extension')
    if extension:
        run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=path / extension)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=('pi3', 'vggt', 'moge', 'mast3r', 'map_anything', 'map_anything_no_ext'))
    parser.add_argument('--root', type=Path,
                        help='External source root (default .depth_dependencies or HAKI_DEPTH_DEPENDENCY_ROOT).')
    parser.add_argument('--check', action='store_true', help='Read-only source/pin/patch check; accepts original-machine fallback checkouts.')
    parser.add_argument('--dry-run', '--dry_run', dest='dry_run', action='store_true', help='Print the source/installation plan without Git/network/pip activity.')
    parser.add_argument('--install', action='store_true', help='After source verification, install dependencies/editable packages in the active Python; preserve Torch/Torchvision.')
    parser.add_argument('--build-curope', action='store_true', help='Explicitly compile the optional CroCo CUDA RoPE extension for MASt3R/Pi3.')
    return parser


def main(argv=None):
    import os

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check and (args.install or args.build_curope):
        parser.error('--check cannot be combined with installation or CUDA build')
    manifest = load_manifest()
    explicit_root = args.root or os.environ.get('HAKI_DEPTH_DEPENDENCY_ROOT')
    source_root = (Path(explicit_root).expanduser().resolve() if explicit_root
                   else PROJECT_ROOT / manifest['default_source_root'])
    names = list(dict.fromkeys(manifest['variant_aliases'].get(name, name)
                              for name in (args.only or manifest['dependencies'])))
    if args.build_curope and 'mast3r' not in names:
        parser.error('--build-curope requires --only mast3r (or the full dependency set)')
    if args.dry_run:
        print(json.dumps({'source_root': str(source_root), 'check_only': args.check,
                          'install': args.install, 'build_curope': args.build_curope,
                          'weights_downloaded': False,
                          'dependencies': [{**manifest['dependencies'][name], 'name': name,
                                            'target': str(source_root / manifest['dependencies'][name]['directory'])}
                                           for name in names]}, indent=2))
        return 0
    try:
        for name in names:
            entry = manifest['dependencies'][name]
            path = (locate_existing(source_root, entry, allow_legacy=not bool(explicit_root))
                    if args.check else source_root / entry['directory'])
            if args.check:
                verify_checkout(path, entry)
            else:
                fetch_checkout(path, entry)
                if args.install:
                    install_checkout(path, entry)
                if args.build_curope:
                    build_curope(path, entry)
            print(f'{name}: pinned source and audited patches verified at {path}')
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Depth dependency setup failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
