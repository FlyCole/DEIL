#!/usr/bin/env python3
"""Interactively view RGB images and masks stored in a Zarr dataset.

The expected project layout is::

    data/img   (steps, cameras, height, width, 3)
    data/mask  (steps, cameras, height, width)

For every step, the viewer shows one row per camera with these columns:

    masked RGB | raw RGB | mask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr


def _load_pyplot():
    """Load Matplotlib with a GUI backend, preferring Tk when needed."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required: pip install matplotlib") from exc

    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    backend = matplotlib.get_backend().lower()
    if backend in noninteractive_backends or "inline" in backend:
        # Tkinter is included in this project's Python environment. Selecting it
        # here avoids requiring MPLBACKEND=TkAgg on every invocation.
        try:
            matplotlib.use("TkAgg", force=True)
        except ImportError as exc:
            raise RuntimeError(
                "Matplotlib cannot open a GUI window from this session. Ensure "
                "that a desktop or forwarded X11/Wayland display is accessible."
            ) from exc

    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if backend in noninteractive_backends or "inline" in backend:
        raise RuntimeError(
            f"Matplotlib could not activate an interactive backend (using {backend!r}). "
            "Run this command in a graphical desktop/X11 session."
        )
    return matplotlib, plt


def _as_uint8_rgb(images: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA image batch to displayable uint8 RGB."""
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None, ...]
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError(
            "Each image step must have shape (cameras, H, W, 3) or (H, W, 3); "
            f"got {images.shape}."
        )

    images = images[..., :3]
    if images.dtype == np.uint8:
        return images

    images = images.astype(np.float32)
    finite = images[np.isfinite(images)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        images = images * 255.0
    return np.nan_to_num(images, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)


def _as_bool_masks(masks: np.ndarray) -> np.ndarray:
    """Convert a mask batch to boolean, treating every nonzero value as foreground."""
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(
            "Each mask step must have shape (cameras, H, W) or (H, W); "
            f"got {masks.shape}."
        )
    return masks != 0


def make_stitched_view(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Build an RGB collage with one camera per row and three labeled columns."""
    images = _as_uint8_rgb(images)
    masks = _as_bool_masks(masks)

    if images.shape[:3] != masks.shape:
        raise ValueError(
            "Image and mask camera/spatial dimensions do not match: "
            f"images={images.shape}, masks={masks.shape}."
        )

    labels = ("MASKED", "RAW", "MASK")
    rows = []
    for camera_index, (image, mask) in enumerate(zip(images, masks)):
        masked = np.where(mask[..., None], image, 0).astype(np.uint8)
        mask_rgb = np.repeat((mask[..., None] * 255).astype(np.uint8), 3, axis=-1)
        panels = [masked, image.copy(), mask_rgb]

        for label, panel in zip(labels, panels):
            text = f"{label}  cam {camera_index}"
            cv2.putText(
                panel,
                text,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                text,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        rows.append(np.concatenate(panels, axis=1))

    return np.concatenate(rows, axis=0)


def view_zarr(
    zarr_path: Path,
    image_key: str = "data/img",
    mask_key: str = "data/mask",
    start: int = 0,
) -> None:
    """Open a Zarr store and show its image/mask steps with Matplotlib."""
    matplotlib, plt = _load_pyplot()
    zarr_path = Path(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    image_key = image_key.strip("/")
    mask_key = mask_key.strip("/")

    try:
        image_array = root[image_key]
        mask_array = root[mask_key]
    except KeyError as exc:
        raise KeyError(
            f"Could not find {exc} in {zarr_path}. "
            "Use --image-key and --mask-key if the arrays have different names."
        ) from exc

    if image_array.ndim not in (4, 5):
        raise ValueError(f"/{image_key} has unsupported shape {image_array.shape}.")
    if mask_array.ndim not in (3, 4, 5):
        raise ValueError(f"/{mask_key} has unsupported shape {mask_array.shape}.")
    if image_array.shape[0] != mask_array.shape[0]:
        raise ValueError(
            f"Step counts differ: /{image_key} has {image_array.shape[0]}, "
            f"/{mask_key} has {mask_array.shape[0]}."
        )

    step_count = int(image_array.shape[0])
    if step_count == 0:
        raise ValueError("The Zarr image array is empty.")
    if not 0 <= start < step_count:
        raise ValueError(f"--start must be between 0 and {step_count - 1}; got {start}.")

    # Matplotlib normally maps q to "close". Remove that binding so q can
    # advance through the Zarr steps instead.
    matplotlib.rcParams["keymap.quit"] = [
        key for key in matplotlib.rcParams["keymap.quit"] if key != "q"
    ]

    print(f"Showing {step_count} steps from {zarr_path}")
    print("Press q for the next step; press Esc to exit.")

    figure, axes = plt.subplots(num=f"Zarr images: {zarr_path.name}")
    axes.axis("off")
    image_artist = axes.imshow(make_stitched_view(image_array[start], mask_array[start]))
    current_step = start

    def show_step(step: int) -> None:
        image_artist.set_data(make_stitched_view(image_array[step], mask_array[step]))
        axes.set_title(f"{zarr_path.name} — step {step + 1}/{step_count}")
        figure.canvas.draw_idle()
        print(f"Step {step + 1}/{step_count}", end="\r", flush=True)

    def on_key(event) -> None:
        nonlocal current_step
        if event.key == "q":
            if current_step == step_count - 1:
                plt.close(figure)
                return
            current_step += 1
            show_step(current_step)
        elif event.key in {"escape", "esc"}:
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    figure.tight_layout()
    show_step(current_step)
    plt.show()
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show masked RGB, raw RGB, and masks from a Zarr dataset."
    )
    parser.add_argument("zarr_path", type=Path, help="Input Zarr store path.")
    parser.add_argument("--image-key", default="data/img", help="RGB array key (default: data/img).")
    parser.add_argument("--mask-key", default="data/mask", help="Mask array key (default: data/mask).")
    parser.add_argument("--start", type=int, default=0, help="Zero-based first step (default: 0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        view_zarr(args.zarr_path, args.image_key, args.mask_key, args.start)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
