"""CPU-only experiment planning for the RLBench Instant Policy evaluator.

Named protocols reproduce the final paper's experiment grids.  ``custom``
provides a separate namespace for experiments with different sample counts.
Importing this module never loads RLBench, PyTorch, or depth estimators.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 1
EVALUATION_TASK_ORDER = (
    'phone_on_base', 'open_box', 'close_box', 'plate_out', 'basketball',
    'close_microwave', 'toilet_seat_down', 'toilet_seat_up',
    'toilet_roll_off', 'meat_off_grill',
)
# These indices are persistent seed identifiers, independent of task filtering.
ALL_TASK_NAMES = (
    'lift_lid', 'phone_on_base', 'open_box', 'slide_block', 'close_box',
    'basketball', 'buzz', 'close_microwave', 'plate_out', 'toilet_seat_down',
    'toilet_seat_up', 'toilet_roll_off', 'open_microwave', 'lamp_on',
    'umbrella_out', 'push_button', 'put_rubbish', 'close_jar', 'open_jar',
    'meat_off_grill',
)
ESTIMATOR_NAMES = (
    'gt', 'pi3', 'vggt', 'moge', 'mast3r', 'map_anything',
    'map_anything_no_ext',
)
CAMERA_SETUP_NAMES = {
    'front': ('front',),
    'front_side': ('front', 'left_shoulder'),
    'front_side_wrist': ('front', 'left_shoulder', 'wrist'),
    'all': ('front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead'),
}
CAMERA_ALIASES = {
    'frc': 'front', 'lsc': 'left_shoulder', 'rsc': 'right_shoulder',
    'wrc': 'wrist', 'ovc': 'overhead',
}
PROTOCOL_DEFAULTS = {
    'screening': dict(num_demos=1, num_demo_iter=5, num_rollouts=2,
                      demo_seed=100_000_000, rollout_seed=200_000_000),
    'paper': dict(num_demos=1, num_demo_iter=20, num_rollouts=10,
                 demo_seed=300_000_000, rollout_seed=400_000_000),
    'custom': dict(num_demos=1, num_demo_iter=1, num_rollouts=5,
                  demo_seed=500_000_000, rollout_seed=600_000_000),
}


@dataclass(frozen=True)
class EvaluationCase:
    task_name: str
    depth_est_name: str
    camera_names: tuple
    resolution: int
    mask_input: bool | None


@dataclass(frozen=True)
class ExperimentConfig:
    protocol: str
    num_demos: int
    num_demo_iter: int
    num_rollouts: int
    demo_seed: int
    rollout_seed: int
    cases: tuple
    experiment_id: str
    demo_cache_dir: str
    rollout_output_dir: str
    headless: bool
    load_cached_demo: bool
    compile_models: bool
    debug: bool
    record_rollouts: bool
    resume: bool
    video_fps: float
    dry_run: bool


def _positive_int(value):
    if isinstance(value, bool):
        raise ValueError('expected a positive integer')
    result = int(value)
    if str(result) != str(value) or result <= 0:
        raise ValueError('expected a positive integer')
    return result


def _seed_value(value):
    if isinstance(value, bool):
        raise ValueError('seed must be an integer between 0 and 4294967295')
    result = int(value)
    if str(result) != str(value) or not 0 <= result <= 2**32 - 1:
        raise ValueError('seed must be an integer between 0 and 4294967295')
    return result


def _positive_float(value):
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError('expected a finite positive number')
    return result


def _arg_type(validator):
    def validate(value):
        try:
            return validator(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return validate


def build_parser():
    parser = argparse.ArgumentParser(
        description='Plan and run RLBench IP screening, Table I, or custom evaluations.',
    )
    parser.add_argument('--protocol', choices=tuple(PROTOCOL_DEFAULTS),
                        help='Experiment protocol (default: screening).')
    parser.add_argument('--config', type=Path, help='JSON experiment configuration; CLI values override it.')
    parser.add_argument('--dry_run', action='store_true', default=None,
                        help='Print the resolved JSON plan without loading models or the simulator.')
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument('--tasks', nargs='+', choices=ALL_TASK_NAMES)
    task_group.add_argument('--task_name', choices=ALL_TASK_NAMES)
    estimator_group = parser.add_mutually_exclusive_group()
    estimator_group.add_argument('--estimators', nargs='+', choices=ESTIMATOR_NAMES)
    estimator_group.add_argument('--depth_est_name', choices=ESTIMATOR_NAMES)
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument('--camera_setups', nargs='+', choices=tuple(CAMERA_SETUP_NAMES))
    camera_group.add_argument('--camera_names', nargs='+',
                              choices=tuple(CAMERA_SETUP_NAMES['all']) + tuple(CAMERA_ALIASES))
    resolution_group = parser.add_mutually_exclusive_group()
    resolution_group.add_argument('--resolutions', nargs='+', type=_arg_type(_positive_int))
    resolution_group.add_argument('--resolution', type=_arg_type(_positive_int))
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument('--mask_input', action='store_true', default=None)
    mask_group.add_argument('--no_mask_input', dest='mask_input', action='store_false')
    for name in ('num_demos', 'num_demo_iter', 'num_rollouts'):
        parser.add_argument('--' + name, type=_arg_type(_positive_int))
    for name in ('demo_seed', 'rollout_seed'):
        parser.add_argument('--' + name, type=_arg_type(_seed_value))
    parser.add_argument('--demo_cache_dir')
    parser.add_argument('--rollout_output_dir')
    for name in ('headless', 'load_cached_demo', 'debug'):
        parser.add_argument('--' + name, action='store_true', default=None)
    parser.add_argument('--compile_models', type=int, choices=(0, 1),
                        help='Compile policy models (0 or 1, default: 0).')
    recording_group = parser.add_mutually_exclusive_group()
    recording_group.add_argument('--record_rollouts', action='store_true', default=None)
    recording_group.add_argument('--no_record_rollouts', dest='record_rollouts', action='store_false')
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument('--resume', action='store_true', default=None)
    resume_group.add_argument('--no_resume', '--no-resume', dest='resume', action='store_false')
    parser.add_argument('--video_fps', type=_arg_type(_positive_float))
    return parser


def _validate_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be a JSON boolean')
    return value


def _sequence(value, name, choices=None):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f'{name} must be a nonempty list')
    if choices is not None and any(item not in choices for item in value):
        raise ValueError(f'invalid {name}; expected values from {list(choices)}')
    if len(value) != len(set(value)):
        raise ValueError(f'{name} contains duplicate values')
    return tuple(value)


def _cameras(value):
    names = _sequence(value, 'camera_names', tuple(CAMERA_SETUP_NAMES['all']) + tuple(CAMERA_ALIASES))
    canonical = tuple(CAMERA_ALIASES.get(name, name) for name in names)
    if len(canonical) != len(set(canonical)):
        raise ValueError('camera_names contains duplicate aliases for the same camera')
    # Camera order is part of a depth adapter's inputs, so preserve explicit order.
    return canonical


def _protocol_cases(protocol):
    cases = []
    for task in EVALUATION_TASK_ORDER:
        if protocol == 'paper':
            for cameras in CAMERA_SETUP_NAMES.values():
                for resolution in (128, 256, 512):
                    cases.append(EvaluationCase(task, 'gt', cameras, resolution, None))
            for estimator in ('pi3', 'vggt'):
                for resolution in (256, 512):
                    for mask in (False, True):
                        cases.append(EvaluationCase(task, estimator, CAMERA_SETUP_NAMES['front_side'], resolution, mask))
                    cases.append(EvaluationCase(task, estimator, CAMERA_SETUP_NAMES['front_side_wrist'], resolution, False))
            for estimator in ('moge', 'mast3r', 'map_anything', 'map_anything_no_ext'):
                cases.append(EvaluationCase(task, estimator, CAMERA_SETUP_NAMES['front_side'], 256, False))
        else:
            for estimator in ESTIMATOR_NAMES[:-1]:
                for cameras in list(CAMERA_SETUP_NAMES.values())[:3]:
                    for resolution in (128, 256, 512):
                        for mask in ((None,) if estimator == 'gt' else (False, True)):
                            cases.append(EvaluationCase(task, estimator, cameras, resolution, mask))
    return tuple(cases)


def _explicit_cases(value):
    if not isinstance(value, list) or not value:
        raise ValueError('cases must be a nonempty list of case objects')
    cases = []
    fields = {'task_name', 'depth_est_name', 'camera_names', 'resolution', 'mask_input'}
    for item in value:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f'each case must have exactly these fields: {sorted(fields)}')
        if item['task_name'] not in ALL_TASK_NAMES:
            raise ValueError(f'unknown task_name: {item["task_name"]}')
        if item['depth_est_name'] not in ESTIMATOR_NAMES:
            raise ValueError(f'unknown depth_est_name: {item["depth_est_name"]}')
        mask = item['mask_input']
        if item['depth_est_name'] == 'gt':
            if mask is not None:
                raise ValueError('GT cases must use mask_input: null')
        else:
            _validate_bool(mask, 'mask_input for learned depth')
        cases.append(EvaluationCase(item['task_name'], item['depth_est_name'],
                                    _cameras(item['camera_names']),
                                    _positive_int(item['resolution']), mask))
    if len(cases) != len(set(cases)):
        raise ValueError('cases contains duplicate configurations')
    return tuple(cases)


def _case_dict(case):
    return {**asdict(case), 'camera_names': list(case.camera_names)}


def _case_sort_key(case):
    tasks = EVALUATION_TASK_ORDER + tuple(name for name in ALL_TASK_NAMES if name not in EVALUATION_TASK_ORDER)
    return (tasks.index(case.task_name), ESTIMATOR_NAMES.index(case.depth_est_name),
            case.camera_names, case.resolution, -1 if case.mask_input is None else int(case.mask_input))


def _selection(data, plural, singular, choices, default=None):
    if plural in data and singular in data:
        raise ValueError(f'{plural} and {singular} are mutually exclusive')
    if plural in data:
        return _sequence(data[plural], plural, choices)
    if singular in data:
        return _sequence([data[singular]], plural, choices)
    return default


def resolve_experiment(args):
    """Validate a CLI/config combination and return its deterministic plan.

    Overrides to a named protocol's sample counts must select ``custom``.  Case
    selectors only filter named grids; they cannot silently invent paper cases.
    """
    cli = vars(args).copy()
    config_path = cli.pop('config', None)
    data = {}
    if config_path is not None:
        try:
            with Path(config_path).open(encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'cannot read experiment config {config_path}: {exc}') from exc
        if not isinstance(data, dict):
            raise ValueError('experiment config must be a JSON object')
    allowed = set(vars(build_parser().parse_args([]))) - {'config'}
    allowed |= {'cases', 'schema_version', 'experiment_id'}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f'unknown config fields: {sorted(unknown)}')
    if data.get('schema_version', SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f'unsupported schema_version; expected {SCHEMA_VERSION}')
    # Replacing one spelling also replaces the alternate spelling from JSON.
    alias_pairs = (('tasks', 'task_name'), ('estimators', 'depth_est_name'),
                   ('resolutions', 'resolution'), ('camera_setups', 'camera_names'))
    for left, right in alias_pairs:
        if cli.get(left) is not None:
            data.pop(right, None)
        if cli.get(right) is not None:
            data.pop(left, None)
    data.update({key: value for key, value in cli.items() if value is not None})
    protocol = data.get('protocol', 'screening')
    if protocol not in PROTOCOL_DEFAULTS:
        raise ValueError(f'unknown protocol: {protocol}')
    sampling = PROTOCOL_DEFAULTS[protocol].copy()
    for name in ('num_demos', 'num_demo_iter', 'num_rollouts'):
        if name in data:
            value = _positive_int(data[name])
            if protocol != 'custom' and value != sampling[name]:
                raise ValueError(f'{protocol} requires {name}={sampling[name]}; use --protocol custom for other counts')
            sampling[name] = value
    for name in ('demo_seed', 'rollout_seed'):
        if name in data:
            sampling[name] = _seed_value(data[name])
    if sampling['demo_seed'] == sampling['rollout_seed']:
        raise ValueError('demo_seed and rollout_seed must differ')
    tasks = _selection(data, 'tasks', 'task_name', ALL_TASK_NAMES)
    estimators = _selection(data, 'estimators', 'depth_est_name', ESTIMATOR_NAMES)
    resolutions = _selection(data, 'resolutions', 'resolution', None)
    if resolutions is not None:
        resolutions = tuple(_positive_int(value) for value in resolutions)
    if 'camera_setups' in data and 'camera_names' in data:
        raise ValueError('camera_setups and camera_names are mutually exclusive')
    cameras = None
    if 'camera_setups' in data:
        cameras = tuple(CAMERA_SETUP_NAMES[name] for name in
                        _sequence(data['camera_setups'], 'camera_setups', CAMERA_SETUP_NAMES))
    elif 'camera_names' in data:
        cameras = (_cameras(data['camera_names']),)
    mask = data.get('mask_input')
    if mask is not None:
        _validate_bool(mask, 'mask_input')
    if 'cases' in data:
        cases = _explicit_cases(data['cases'])
        if protocol != 'custom':
            allowed_cases = set(_protocol_cases(protocol))
            if any(case not in allowed_cases for case in cases):
                raise ValueError(f'cases contains configurations outside {protocol}; use --protocol custom')
    elif protocol == 'custom':
        cases = tuple(
            EvaluationCase(task, estimator, camera, resolution, masked)
            for task in (tasks or EVALUATION_TASK_ORDER)
            for estimator in (estimators or ESTIMATOR_NAMES[:-1])
            for camera in (cameras or (CAMERA_SETUP_NAMES['front_side'],))
            for resolution in (resolutions or (256,))
            for masked in ((None,) if estimator == 'gt' else ((False, True) if mask is None else (mask,)))
        )
    else:
        cases = _protocol_cases(protocol)
    selected = tuple(case for case in cases
                     if (tasks is None or case.task_name in tasks)
                     and (estimators is None or case.depth_est_name in estimators)
                     and (resolutions is None or case.resolution in resolutions)
                     and (cameras is None or case.camera_names in cameras)
                     and (mask is None or case.mask_input is None or case.mask_input == mask))
    if not selected:
        raise ValueError(f'no cases match the requested {protocol} selectors')
    # Catch selectors that would silently omit requested values from a named grid.
    for requested, represented, name in (
        (tasks, {case.task_name for case in selected}, 'tasks'),
        (estimators, {case.depth_est_name for case in selected}, 'estimators'),
        (resolutions, {case.resolution for case in selected}, 'resolutions'),
        (cameras, {case.camera_names for case in selected}, 'camera selections'),
    ):
        if requested is not None and any(item not in represented for item in requested):
            raise ValueError(f'some requested {name} have no matching {protocol} cases')
    selected = tuple(sorted(selected, key=_case_sort_key))
    for task in {case.task_name for case in selected}:
        evaluation_seed(sampling['demo_seed'], task, sampling['num_demo_iter'] - 1,
                        sampling['num_demos'] - 1)
        evaluation_seed(sampling['rollout_seed'], task, sampling['num_demo_iter'] - 1,
                        sampling['num_rollouts'] - 1)
    identity = dict(schema_version=SCHEMA_VERSION, protocol=protocol, **sampling,
                    cases=[_case_dict(case) for case in selected])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    experiment_id = f'{protocol}-v{SCHEMA_VERSION}-{digest}'
    flags = {name: _validate_bool(data.get(name, default), name)
             for name, default in (('headless', False), ('load_cached_demo', False),
                                   ('debug', False), ('record_rollouts', True),
                                   ('dry_run', False))}
    compile_models = data.get('compile_models', 0)
    if compile_models not in (False, True, 0, 1):
        raise ValueError('compile_models must be 0 or 1')
    resume = data.get('resume')
    resume = flags['record_rollouts'] if resume is None else _validate_bool(resume, 'resume')
    paths = {}
    for name, default in (('demo_cache_dir', 'demos_ip'), ('rollout_output_dir', 'results/ip_eval')):
        value = data.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty path string')
        paths[name] = value
    return ExperimentConfig(protocol=protocol, **sampling, cases=selected,
                            experiment_id=experiment_id, **paths, **flags,
                            compile_models=bool(compile_models), resume=resume,
                            video_fps=_positive_float(data.get('video_fps', 10.0)))


def experiment_to_dict(config):
    """Return a JSON serializable plan suitable for result provenance."""
    result = asdict(config)
    result['schema_version'] = SCHEMA_VERSION
    result['cases'] = [_case_dict(case) for case in config.cases]
    return result


def evaluation_seed(base_seed, task_name, demo_iter, rollout_index=0):
    """Map a task/context/rollout tuple to a stable, collision-free 32-bit seed.

    Cantor pairing avoids Python's randomized ``hash()`` and keeps each task's
    seed unchanged when other tasks or cases are filtered out. Named protocols
    use distinct base ranges for demonstrations and rollouts.
    """
    base_seed = _seed_value(base_seed)
    if task_name not in ALL_TASK_NAMES:
        raise ValueError(f'unknown task_name: {task_name}')
    for value in (demo_iter, rollout_index):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('demo_iter and rollout_index must be nonnegative integers')
    def pair(left, right):
        total = left + right
        return total * (total + 1) // 2 + right
    offset = pair(ALL_TASK_NAMES.index(task_name), pair(demo_iter, rollout_index))
    result = base_seed + offset
    if result > 2**32 - 1:
        raise ValueError('experiment indices and base seed exceed the 32-bit seed range')
    return result


if __name__ == '__main__':
    parser = build_parser()
    try:
        experiment = resolve_experiment(parser.parse_args())
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(experiment_to_dict(experiment), indent=2))
