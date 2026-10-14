#!/usr/bin/env python3
"""Fetch the pinned PyRep/RLBench SDK and preserve the local demo protocol.

Default operation only fetches sources into .runtime_dependencies and applies
an audited RLBench patch. --check is read-only. --check-installed verifies the
active Python's SDK source without importing the simulator. --install explicitly
installs PyRep before RLBench and requires COPPELIASIM_ROOT; pinned PyRep setup
may append allowOldEduRelease to CoppeliaSim's system/usrset.txt.

The patch preserves known legacy failure handling. It does not fix infinite
reset retries or the random-EEF dummy cleanup; see runtime_dependencies.json.
"""

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / 'runtime_dependencies.json'


def common_helpers():
    # Environments may install an unrelated ``scripts`` package. Load the
    # shared stdlib source verifier by filename under a private alias.
    spec = importlib.util.spec_from_file_location(
        '_haki_runtime_source_helpers', PROJECT_ROOT / 'scripts' / 'bootstrap_depth_dependencies.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest():
    with MANIFEST_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


def verify_api(path, entry):
    specification = entry.get('required_api')
    if not specification:
        return
    tree = ast.parse((path / specification['file']).read_text())
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == specification['class']), None)
    method = None if cls is None else next(
        (node for node in cls.body if isinstance(node, ast.FunctionDef)
         and node.name == specification['method']), None
    )
    parameters = [] if method is None else [arg.arg for arg in method.args.posonlyargs
                                           + method.args.args + method.args.kwonlyargs]
    if specification['parameter'] not in parameters:
        raise ValueError(
            f'{path}: required {specification["class"]}.{specification["method"]}'
            f' parameter {specification["parameter"]} is missing; apply the recorded local SDK patch'
        )


def verify_source(path, entry, helpers):
    helpers.verify_checkout(path, entry)
    license_info = entry['license']
    if helpers.sha256(path / license_info['file']) != license_info['sha256']:
        raise ValueError(f'Pinned upstream license content does not match: {path}')
    verify_api(path, entry)


def verify_installed(entry, helpers):
    distribution = importlib.metadata.distribution(entry['distribution'])
    if distribution.version != entry['version']:
        raise ValueError(
            f'{entry["distribution"]}: installed version {distribution.version} '
            f'does not match {entry["version"]}'
        )
    # Top-level find_spec resolves paths but does not execute pyrep/rlbench,
    # so no native CoppeliaSim library or simulator process is loaded here.
    spec = importlib.util.find_spec(entry['package'])
    if spec is None or not spec.origin:
        raise ValueError(f'Cannot locate installed {entry["package"]} source')
    root = Path(spec.origin).resolve().parent.parent
    for filename, expected in entry['installed_python_files'].items():
        source = root / filename
        if not source.is_file() or helpers.sha256(source) != expected:
            raise ValueError(
                f'Installed SDK source differs from the pinned audited version: {source}. '
                'Install the prepared checkout with --install in the intended environment.'
            )
    verify_api(root, entry)
    return root


def install_sources(paths, names, manifest, helpers):
    configured = os.environ.get('COPPELIASIM_ROOT')
    if not configured:
        raise ValueError('--install requires COPPELIASIM_ROOT for the chosen CoppeliaSim 4.1 installation')
    simulator = Path(configured).expanduser().resolve()
    if not simulator.is_dir() or not (simulator / 'libcoppeliaSim.so').is_file():
        raise ValueError(f'COPPELIASIM_ROOT does not contain libcoppeliaSim.so: {simulator}')
    print('Explicit SDK installation may update CoppeliaSim system/usrset.txt via upstream PyRep setup.')
    with tempfile.TemporaryDirectory(prefix='haki-runtime-pip-') as directory:
        constraints = Path(directory) / 'numpy.txt'
        constraints.write_text('numpy' + manifest['install_requirements']['numpy_constraint'] + '\n')
        for name in manifest['installation_order']:
            if name in names:
                requirements = manifest['dependencies'][name]['runtime_requirements']
                if requirements:
                    helpers.run([sys.executable, '-m', 'pip', 'install', '-c', str(constraints),
                                 *requirements], cwd=paths[name])
                # RLBench's pinned setup.py refers to unpinned upstream PyRep.
                # Install declared non-SDK dependencies separately, then keep
                # the explicitly prepared PyRep checkout with --no-deps.
                helpers.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                             '--no-deps', '-c', str(constraints), '-e', str(paths[name])], cwd=paths[name])


def selected_names(requested, manifest):
    names = set(requested or manifest['dependencies'])
    for name in tuple(names):
        names.update(manifest['dependencies'][name].get('requires', []))
    return [name for name in manifest['installation_order'] if name in names]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=('pyrep', 'rlbench'))
    parser.add_argument('--root', type=Path,
                        help='Source root; default .runtime_dependencies or HAKI_RUNTIME_DEPENDENCY_ROOT.')
    parser.add_argument('--check', action='store_true', help='Read-only source commit, patch, license and API verification.')
    parser.add_argument('--check-installed', action='store_true', help='Read-only active SDK source hashes/API check; no simulator imports.')
    parser.add_argument('--dry-run', '--dry_run', dest='dry_run', action='store_true')
    parser.add_argument('--install', action='store_true', help='Explicitly install PyRep then RLBench; requires COPPELIASIM_ROOT and can update its usrset.txt.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.install and (args.check or args.check_installed):
        parser.error('Read-only checks cannot be combined with --install')
    manifest = load_manifest()
    configured_root = args.root or os.environ.get('HAKI_RUNTIME_DEPENDENCY_ROOT')
    root = (Path(configured_root).expanduser().resolve() if configured_root
            else PROJECT_ROOT / manifest['default_source_root'])
    names = selected_names(args.only, manifest)
    if args.dry_run:
        print(json.dumps({
            'source_root': str(root), 'check': args.check, 'check_installed': args.check_installed,
            'install': args.install, 'installation_order': names,
            'legacy_failure_handling_preserved': True,
            'dependencies': [
                {key: manifest['dependencies'][name][key] for key in
                 ('repository', 'commit', 'directory', 'patches', 'license')}
                for name in names
            ],
            'install_requirements': manifest['install_requirements'],
        }, indent=2))
        return 0
    helpers = common_helpers()
    try:
        if args.check_installed:
            for name in names:
                active = verify_installed(manifest['dependencies'][name], helpers)
                print(f'{name}: active installed Python sources/API match at {active}')
            if not args.check:
                return 0
        paths = {}
        for name in names:
            entry = manifest['dependencies'][name]
            path = root / entry['directory']
            if not args.check:
                helpers.fetch_checkout(path, entry)
            verify_source(path, entry, helpers)
            paths[name] = path
            print(f'{name}: pinned SDK source/patch/license/API verified at {path}')
        if args.install:
            install_sources(paths, names, manifest, helpers)
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError, helpers.subprocess.CalledProcessError) as error:
        parser.exit(1, f'Runtime dependency setup failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
