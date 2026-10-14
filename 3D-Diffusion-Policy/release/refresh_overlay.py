#!/usr/bin/env python3
"""Refresh the reviewable DP3 overlay from an existing upstream worktree.

This reads Git objects and diffs; it never stages, commits, or edits .git.
"""

import hashlib
import json
from pathlib import Path
import stat
import subprocess


ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"
UPSTREAM = "https://github.com/YanjieZe/3D-Diffusion-Policy.git"
COMMIT = "47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97"
EXCLUDED_PARTS = {".git", "release", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "data", "checkpoints", "media", "outputs", "output", "build", "dist", ".venv", "venv", ".eggs", "test", "tests"}
EXCLUDED_FILE_PATTERNS = ["test_*.py", "*_test.py", "test.py", "tests.py", "test_requirements.txt"]
SOURCE_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml", ".xml", ".txt", ".md", ".rst", ".sh", ".bash"}
SOURCE_NAMES = {".gitignore", ".gitattributes", "MANIFEST.in", "Makefile", "Dockerfile"}


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def _is_test(name):
    path = Path(name)
    # gym/envs/unittest contains registered environments, not a test suite.
    return (
        any(part in {"test", "tests"} for part in path.parts)
        or (path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py")))
        or path.name in {"test.py", "tests.py", "test_requirements.txt"}
    )


def _is_source(name):
    path = Path(name)
    if _is_test(name) or any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts):
        return False
    return path.suffix.lower() in SOURCE_SUFFIXES or path.name in SOURCE_NAMES or path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))


def _local_sources(directory):
    sources = {}
    for path in directory.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"Symlinks require an explicit release decision: {path}")
        relative = path.relative_to(ROOT).as_posix()
        if path.is_dir():
            if not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts):
                sources.update(_local_sources(path))
        elif _is_source(relative):
            sources[relative] = path
    return sources


def main():
    baseline = {}
    for record in _git("ls-tree", "-r", "-z", COMMIT).split(b"\0"):
        if not record:
            continue
        metadata, encoded_name = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        name = encoded_name.decode()
        if _is_source(name):
            if kind != "blob" or mode not in ("100644", "100755"):
                raise RuntimeError(f"Unsupported source tree entry: {name}")
            baseline[name] = {"mode": mode, "object_id": object_id}

    local = _local_sources(ROOT)
    baseline_bytes = {}
    process = subprocess.Popen(["git", "-C", str(ROOT), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for name, item in baseline.items():
            process.stdin.write((item["object_id"] + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            content = process.stdout.read(int(header[2]))
            process.stdout.read(1)
            baseline_bytes[name] = content
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError("Cannot read DP3 upstream Git objects")

    added = sorted(set(local) - set(baseline))
    deleted = sorted(set(baseline) - set(local))
    modified = sorted(name for name in set(local) & set(baseline) if local[name].read_bytes() != baseline_bytes[name])
    tracked_changes = _git("diff", "--name-only", "-z", COMMIT).split(b"\0")
    unsupported_changes = [name.decode() for name in tracked_changes if name and not _is_source(name.decode()) and not _is_test(name.decode())]
    if unsupported_changes:
        raise RuntimeError("Non-source upstream changes require an explicit release decision: " + repr(unsupported_changes))

    # Tests are omitted from the release snapshot rather than changed in the
    # scientific overlay; restrict the diff to the selected source paths.
    patch = _git("diff", "--binary", "--full-index", "--no-ext-diff", COMMIT, "--", *sorted(set(baseline) | set(local)))
    for name in added:
        result = subprocess.run(["git", "diff", "--no-index", "--binary", "--full-index", "--", "/dev/null", name], cwd=ROOT, capture_output=True)
        if result.returncode != 1:
            raise RuntimeError(f"Cannot create new-file patch for {name}: {result.stderr.decode()}")
        patch += result.stdout
    (RELEASE / "overlay.patch").write_bytes(patch)

    files = {}
    for name, path in sorted(local.items()):
        content = path.read_bytes()
        files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "mode": "0755" if stat.S_IMODE(path.stat().st_mode) & stat.S_IXUSR else "0644",
        }
    identity = "".join(name + "\0" + item["sha256"] + "\n" for name, item in files.items()).encode()
    manifest = {
        "schema_version": 1,
        "upstream_repository": UPSTREAM,
        "upstream_commit": COMMIT,
        "source_sha256": hashlib.sha256(identity).hexdigest(),
        "patch": {"path": "overlay.patch", "sha256": hashlib.sha256(patch).hexdigest(), "size_bytes": len(patch)},
        "source_files": files,
        "deleted_upstream_files": {
            name: {"sha256": hashlib.sha256(baseline_bytes[name]).hexdigest()} for name in deleted
        },
        "local_changes": {"modified": modified, "added": added, "deleted": deleted},
        "excluded_directory_names": sorted(EXCLUDED_PARTS),
        "excluded_file_patterns": EXCLUDED_FILE_PATTERNS,
        "licenses": {name: files[name]["sha256"] for name in files if Path(name).name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))},
    }
    (RELEASE / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_files": len(files), "modified": len(modified), "added": len(added), "deleted": len(deleted), "patch_bytes": len(patch), "source_sha256": manifest["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
