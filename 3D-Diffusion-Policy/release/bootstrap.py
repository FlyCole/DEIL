#!/usr/bin/env python3
"""Restore the pinned DP3 sources and local changes without replacing a checkout."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile


RELEASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = RELEASE_DIR.parent


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or ".git" in path.parts:
        raise ValueError(f"Unsafe source manifest path: {value}")
    return path


def _manifest(manifest_path=None):
    path = RELEASE_DIR / "source_manifest.json" if manifest_path is None else Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported DP3 source manifest schema")
    for name in (*manifest["source_files"], *manifest["deleted_upstream_files"]):
        _relative_path(name)
    return manifest


def _has_symlink_parent(root, name):
    parts = _relative_path(name).parts
    return any(root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts)))


def verify_source(source_root=None, manifest_path=None):
    """Return a stdlib-only integrity report; no network, model or simulator import."""
    root = DEFAULT_SOURCE_ROOT if source_root is None else Path(source_root)
    manifest = _manifest(manifest_path)
    missing, modified = [], []
    for name, expected in manifest["source_files"].items():
        path = root / name
        if _has_symlink_parent(root, name):
            modified.append(name)
        elif path.is_symlink() or not path.is_file():
            missing.append(name)
        elif _sha256(path) != expected["sha256"]:
            modified.append(name)
    unexpected_deleted = [name for name in manifest["deleted_upstream_files"] if (root / name).exists()]
    return {
        "ok": not missing and not modified and not unexpected_deleted,
        "source_root": str(root.resolve()),
        "upstream_commit": manifest["upstream_commit"],
        "source_sha256": manifest["source_sha256"],
        "files_checked": len(manifest["source_files"]),
        "missing": missing,
        "modified": modified,
        "unexpected_deleted": unexpected_deleted,
    }


def _run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def _source_conflicts(root, manifest):
    conflicts = []
    for name, expected in manifest["source_files"].items():
        path = root / name
        if _has_symlink_parent(root, name):
            conflicts.append(name)
        elif path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or _sha256(path) != expected["sha256"]:
                conflicts.append(name)
    for name, expected in manifest["deleted_upstream_files"].items():
        path = root / name
        if _has_symlink_parent(root, name):
            conflicts.append(name)
        elif path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or _sha256(path) != expected["sha256"]:
                conflicts.append(name)
    return conflicts


def bootstrap_source(source_root=None, upstream_source=None):
    """Fill missing source files after verifying a restored isolated upstream tree.

    Existing source content must already match the release. Data, weights,
    outputs, installed packages and the target's .git directory are untouched.
    """
    root = DEFAULT_SOURCE_ROOT if source_root is None else Path(source_root).resolve()
    manifest = _manifest()
    report = verify_source(root)
    if report["ok"]:
        return report
    conflicts = _source_conflicts(root, manifest)
    if conflicts:
        raise RuntimeError("DP3 source differs from the release; preserving existing files:\n" + "\n".join(conflicts))
    patch = RELEASE_DIR / manifest["patch"]["path"]
    if _sha256(patch) != manifest["patch"]["sha256"]:
        raise RuntimeError("DP3 overlay checksum differs from the release manifest")

    source = manifest["upstream_repository"] if upstream_source is None else str(upstream_source)
    with tempfile.TemporaryDirectory(prefix="haki-dp3-source-") as temporary:
        restored = Path(temporary) / "upstream"
        restored.mkdir()
        _run("git", "init", "--quiet", str(restored))
        _run("git", "-C", str(restored), "fetch", "--quiet", "--depth", "1", source, manifest["upstream_commit"])
        actual_commit = subprocess.check_output(
            ["git", "-C", str(restored), "rev-parse", "FETCH_HEAD"], text=True
        ).strip()
        if actual_commit != manifest["upstream_commit"]:
            raise RuntimeError("Fetched DP3 commit differs from the pinned upstream revision")
        _run("git", "-C", str(restored), "checkout", "--quiet", "--detach", actual_commit)
        _run("git", "-C", str(restored), "apply", "--check", "--whitespace=nowarn", str(patch))
        _run("git", "-C", str(restored), "apply", "--whitespace=nowarn", str(patch))
        restored_report = verify_source(restored)
        if not restored_report["ok"]:
            raise RuntimeError("Restored DP3 snapshot failed integrity checks: " + json.dumps(restored_report))

        # Recheck after fetching before making changes, so edits made during the
        # download are also preserved. Only known release sources are copied.
        conflicts = _source_conflicts(root, manifest)
        if conflicts:
            raise RuntimeError("DP3 source changed during reconstruction; preserving existing files:\n" + "\n".join(conflicts))
        for name, expected in manifest["source_files"].items():
            destination = root / name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(restored / name, destination)
                destination.chmod(int(expected["mode"], 8))
        for name in manifest["deleted_upstream_files"]:
            path = root / name
            if path.is_file():
                path.unlink()
    return verify_source(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--upstream-source", help="optional Git URL or local Git checkout containing the pinned commit")
    parser.add_argument("--check", action="store_true", help="verify existing source without fetching or changing files")
    args = parser.parse_args(argv)
    try:
        report = verify_source(args.target_dir) if args.check else bootstrap_source(args.target_dir, args.upstream_source)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"DP3 source preparation failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
