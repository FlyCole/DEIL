"""Compatibility helpers for the pinned, publicly released Instant Policy.

The public model is a native extension with a deployment API, rather than the
private ``ip`` training package. Importing this adapter requires only stdlib;
model and geometry dependencies are loaded when the corresponding API is used.
"""

from functools import lru_cache
import hashlib
import importlib.util
import inspect
import operator
from pathlib import Path
import sys


UPSTREAM_REPOSITORY = "https://github.com/vv19/instant_policy"
UPSTREAM_COMMIT = "72281736672a555e9c9834eb15c225e2f1baaf93"
NATIVE_MODULE_SHA256 = "ef65547a8cce3699dbea7917a64dbee443a812f0e949bec9ef545c7974bcecc7"
PUBLIC_UTILS_SHA256 = "c6b0c0a59f4c17d3767a47003a18ceca5cd33a68c8ca5533b6256008bdfbab31"
CHECKPOINT_SOURCE = "https://drive.google.com/drive/folders/1hfyQ0DhZ8sCLrrH7dmE4WLMIibxZGVpI"
_VENDOR_DIR = Path(__file__).resolve().parent / "instant_policy"
_UTILS_MODULE_NAME = "_haki_public_instant_policy_utils"
_PUBLIC_NATIVE_MODULE = None
_PUBLIC_NATIVE_PATH = None
_ABI_HELP = (
    "The published upstream environment uses Python 3.10 and PyTorch 2.2.0 "
    "with CUDA 11.8. RTX 50-series GPUs require a newer supported stack; "
    "Python 3.10/PyTorch 2.7.0/CUDA 12.8 passed the public checkpoint smoke "
    "check on RTX 5090. See README.md for matching PyG dependencies and "
    "hardware-specific setup. The private ip package is not a fallback."
)


@lru_cache(maxsize=16)
def _cached_sha256(resolved_path, size_bytes, mtime_ns):
    # Size and modification time are cache keys, so a replaced checkpoint is
    # hashed again while repeated experiment manifests avoid rereading it.
    digest = hashlib.sha256()
    with open(resolved_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_sha256(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _cached_sha256(str(resolved), stat.st_size, stat.st_mtime_ns)


def _checkpoint_path(checkpoint_path):
    path = _VENDOR_DIR / "checkpoints" / "model.pt" if checkpoint_path is None else Path(checkpoint_path).expanduser()
    return path.resolve()


def public_policy_provenance(checkpoint_path=None):
    """Describe the pinned release and checkpoint without loading the model.

    A missing checkpoint is valid for command previews. The checkpoint SHA-256
    identifies weights actually present locally; upstream publishes no fixed
    checksum for its download folder.
    """
    checkpoint = _checkpoint_path(checkpoint_path)
    exists = checkpoint.is_file()
    return {
        "backend": "instant_policy_public",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "native_module": "instant_policy/instant_policy.so",
        "native_module_sha256": NATIVE_MODULE_SHA256,
        "checkpoint": {
            "path": str(checkpoint),
            "source": CHECKPOINT_SOURCE if checkpoint_path is None else "user-supplied local checkpoint",
            "exists": exists,
            "size_bytes": checkpoint.stat().st_size if exists else None,
            "sha256": _file_sha256(checkpoint) if exists else None,
        },
    }


def _validate_public_api(module):
    graph = getattr(module, "GraphDiffusion", None)
    if not callable(getattr(graph, "load_from_checkpoint", None)) or not callable(
        getattr(module, "sample_to_cond_demo", None)
    ):
        raise RuntimeError(
            "instant_policy does not expose the public GraphDiffusion deployment API. "
            "A namespace directory or private ip module cannot provide this backend."
        )
    return module


def _load_public_module():
    """Load the exact extension by filename, using its required native name."""
    global _PUBLIC_NATIVE_MODULE, _PUBLIC_NATIVE_PATH

    native_path = (_VENDOR_DIR / "instant_policy.so").resolve()
    existing = sys.modules.get("instant_policy")
    if "instant_policy" in sys.modules:
        # Nuitka rewrites both __file__ and the spec to a generated .py path.
        # Trust that object only after this adapter loaded the verified .so.
        origin = getattr(getattr(existing, "__spec__", None), "origin", None) or getattr(existing, "__file__", None)
        verified_object = existing is _PUBLIC_NATIVE_MODULE and _PUBLIC_NATIVE_PATH == native_path
        if not verified_object and (origin is None or Path(origin).resolve() != native_path):
            raise RuntimeError(
                "The loaded instant_policy module is a namespace or a different implementation. "
                "Restart with the pinned public instant_policy/instant_policy.so; "
                "the private ip package cannot be used."
            )

    if not native_path.is_file():
        raise RuntimeError(
            f"Public Instant Policy extension not found: {native_path}. "
            f"Restore the vendored release at commit {UPSTREAM_COMMIT}."
        )
    if _file_sha256(native_path) != NATIVE_MODULE_SHA256:
        raise RuntimeError(
            f"Public Instant Policy extension checksum differs from commit {UPSTREAM_COMMIT}: "
            f"{native_path}. Restore the pinned public release."
        )
    if existing is not None:
        return _validate_public_api(existing)

    module = None
    try:
        # The extension exports PyInit_instant_policy. A distinct alias would
        # fail native initialization, unlike the pure Python utility module.
        spec = importlib.util.spec_from_file_location("instant_policy", native_path)
        if spec is None or spec.loader is None:
            raise ImportError("No native extension loader is available")
        module = importlib.util.module_from_spec(spec)
        sys.modules["instant_policy"] = module
        spec.loader.exec_module(module)
        public = _validate_public_api(module)
        _PUBLIC_NATIVE_MODULE = public
        _PUBLIC_NATIVE_PATH = native_path
        return public
    except Exception as exc:
        if module is not None and sys.modules.get("instant_policy") is module:
            del sys.modules["instant_policy"]
        raise RuntimeError(f"Cannot load public Instant Policy: {exc}. {_ABI_HELP}") from exc


def _positive_integer(value, name):
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def load_public_policy(num_demos=1, device="cuda", checkpoint_path=None, num_diffusion_steps=4):
    """Load official weights using only the public deployment methods."""
    num_demos = _positive_integer(num_demos, "num_demos")
    num_diffusion_steps = _positive_integer(num_diffusion_steps, "num_diffusion_steps")
    checkpoint = _checkpoint_path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Public Instant Policy checkpoint not found: {checkpoint}. "
            "Run instant_policy/download_weights.sh to obtain the upstream model.pt."
        )
    public = _load_public_module()
    model = public.GraphDiffusion.load_from_checkpoint(
        str(checkpoint), device=device, strict=True, map_location=device
    )
    model.set_num_demos(num_demos)
    model.set_num_diffusion_steps(num_diffusion_steps)
    model.eval()
    return model


def sample_to_cond_demo(sample, num_waypoints=10, num_points=2048):
    """Adapt the public deployment/stub signatures without private preprocessing."""
    num_waypoints = _positive_integer(num_waypoints, "num_waypoints")
    num_points = _positive_integer(num_points, "num_points")
    preprocess = _load_public_module().sample_to_cond_demo
    try:
        signature = inspect.signature(preprocess)
    except (TypeError, ValueError):
        # Upstream deploy.py uses two arguments and relies on the 2048-point
        # default. Some native callables do not expose an inspectable signature.
        if num_points == 2048:
            return preprocess(sample, num_waypoints)
        return preprocess(sample, num_waypoints, num_points)

    try:
        signature.bind(sample, num_waypoints, num_points)
    except TypeError:
        try:
            signature.bind(sample, num_waypoints, num_points=num_points)
        except TypeError:
            if num_points != 2048:
                raise ValueError("This public sample_to_cond_demo API supports only the default 2048 points")
            signature.bind(sample, num_waypoints)
            return preprocess(sample, num_waypoints)
        return preprocess(sample, num_waypoints, num_points=num_points)
    return preprocess(sample, num_waypoints, num_points)


def _numpy_array(value):
    import numpy as np

    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


def predict_actions(model, full_sample):
    """Return finite public relative transforms [H,4,4] and gripper values [H]."""
    import numpy as np

    predictor = getattr(model, "predict_actions", None)
    if not callable(predictor):
        raise RuntimeError("The policy must expose the public predict_actions(full_sample) API")
    prediction = predictor(full_sample)
    if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
        raise ValueError("Public predict_actions must return (actions, grips)")
    actions, grips = (_numpy_array(value) for value in prediction)
    if actions.ndim != 3 or actions.shape[1:] != (4, 4) or actions.shape[0] < 1:
        raise ValueError(f"Public actions must have shape [H,4,4] with H > 0, got {actions.shape}")
    horizon = actions.shape[0]
    if grips.shape not in ((horizon,), (horizon, 1)):
        raise ValueError(f"Public grips must have shape [H] or [H,1], got {grips.shape}")
    try:
        finite = np.isfinite(actions).all() and np.isfinite(grips).all()
    except TypeError as exc:
        raise ValueError("Public actions and grips must be numeric arrays") from exc
    if not finite:
        raise ValueError("Public actions and grips must contain only finite values")
    return actions, grips.reshape(horizon)


def _load_public_utils():
    """Load public geometry helpers under an alias that cannot shadow project utils."""
    public_path = (_VENDOR_DIR / "utils.py").resolve()
    if not public_path.is_file() or _file_sha256(public_path) != PUBLIC_UTILS_SHA256:
        raise RuntimeError(f"Pinned public Instant Policy utilities are missing or modified: {public_path}")
    existing = sys.modules.get(_UTILS_MODULE_NAME)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != public_path:
            raise RuntimeError("The public Instant Policy utility alias refers to a different file")
        return existing
    module = None
    try:
        spec = importlib.util.spec_from_file_location(_UTILS_MODULE_NAME, public_path)
        if spec is None or spec.loader is None:
            raise ImportError("No Python utility loader is available")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_UTILS_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        if module is not None and sys.modules.get(_UTILS_MODULE_NAME) is module:
            del sys.modules[_UTILS_MODULE_NAME]
        raise RuntimeError(
            f"Cannot load public Instant Policy geometry helpers: {exc}. "
            "Install their NumPy, SciPy and Open3D runtime dependencies."
        ) from exc


def subsample_pcd(sample, num_points=2048):
    return _load_public_utils().subsample_pcd(sample, _positive_integer(num_points, "num_points"))


def pose_to_transform(pose):
    return _load_public_utils().pose_to_transform(pose)


def transform_to_pose(transform):
    return _load_public_utils().transform_to_pose(transform)


def transform_pcd(points, transform):
    return _load_public_utils().transform_pcd(points, transform)
