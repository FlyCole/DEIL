"""Collect the raw RLBench demonstrations used by the main IP evaluator.

The planner is shared with ``rlb_task_eval.py``. Each task/camera/resolution
combination is collected once, regardless of the number of depth methods or
masking settings in the experiment. ``--dry_run`` needs only Python's standard
library and never loads the simulator or policy weights.
"""

import json
from pathlib import Path

from rlb_eval_config import (
    build_parser,
    evaluation_seed,
    resolve_experiment,
)


def collection_cases(experiment):
    """Keep one collection case for each shared raw observation configuration."""
    seen = set()
    cases = []
    for case in experiment.cases:
        key = (case.task_name, case.camera_names, case.resolution)
        if key not in seen:
            seen.add(key)
            cases.append(key)
    return cases


def collection_plan(experiment):
    cases = collection_cases(experiment)
    return {
        'protocol': experiment.protocol,
        'demo_seed': experiment.demo_seed,
        'num_demos': experiment.num_demos,
        'num_demo_iter': experiment.num_demo_iter,
        'demo_cache_dir': experiment.demo_cache_dir,
        'headless': experiment.headless,
        'num_cache_files': len(cases) * experiment.num_demo_iter,
        'cases': [
            {'task_name': task, 'camera_names': list(cameras), 'resolution': resolution}
            for task, cameras, resolution in cases
        ],
    }


def collect_demonstrations(experiment):
    from rlb_task_eval import (
        demo_cache_path,
        get_demos,
        initialise_rlbench,
        load_raw_demo_cache,
        save_raw_demo_cache,
        set_random_seed,
    )

    for task_name, camera_names, resolution in collection_cases(experiment):
        for demo_iter in range(experiment.num_demo_iter):
            metadata = {
                'protocol': experiment.protocol,
                'demo_seed': experiment.demo_seed,
                'task_name': task_name,
                'camera_names': list(camera_names),
                'resolution': resolution,
                'demo_iter': demo_iter,
                'num_demos': experiment.num_demos,
            }
            path = demo_cache_path(experiment.demo_cache_dir, **metadata)
            if Path(path).exists():
                load_raw_demo_cache(path, metadata)
                print(f'Using existing validated demo cache: {path}')
                continue

            seed = evaluation_seed(experiment.demo_seed, task_name, demo_iter)
            set_random_seed(seed)
            env, task = initialise_rlbench(
                task_name=task_name,
                camera_names=list(camera_names),
                resolution=resolution,
                restrict_rot=True,
                headless=experiment.headless,
            )
            try:
                set_random_seed(seed)
                full_sample = get_demos(task, experiment.num_demos)
                save_raw_demo_cache(path, full_sample, metadata)
                print(f'Saved demo cache: {path}')
            finally:
                env.shutdown()


def main(argv=None):
    parser = build_parser()
    parser.description = 'Collect reusable raw RLBench demonstrations for the IP experiment protocol.'
    try:
        experiment = resolve_experiment(parser.parse_args(argv))
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(collection_plan(experiment), indent=2))
    if not experiment.dry_run:
        collect_demonstrations(experiment)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
