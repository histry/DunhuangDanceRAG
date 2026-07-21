"""Lazy scheduler facade for the optional learned transition sampler.

The training implementation has a large dependency closure. Keeping the import
behind these functions lets the default deterministic scheduler start without
loading contact-INR and boundary-training modules.
"""
from __future__ import annotations

from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    convert_motion_rot6d_layout_np,
    normalize_rot6d_layout,
)


def load_transition_diffusion(*args, fps=None, **kwargs):
    from training.transition_diffusion import load_transition_diffusion as implementation

    bundle = implementation(*args, fps=fps, **kwargs)
    if bundle is None:
        return None
    config = bundle.get("config", {})
    layout = normalize_rot6d_layout(
        config.get("rot6d_layout", ROT6D_LAYOUT_PYTORCH3D_ROW)
    )
    if layout != ROT6D_LAYOUT_PYTORCH3D_ROW:
        raise RuntimeError(
            "The optional historical transition diffusion implementation is "
            f"PyTorch3D-row native, but checkpoint declares {layout!r}."
        )
    bundle["rot6d_layout"] = layout
    bundle["canonical_rot6d_layout"] = CANONICAL_ROT6D_LAYOUT
    return bundle


def sample_transition_diffusion(
    bundle,
    start_frame,
    end_frame,
    length,
    music_query,
    rough=None,
    device="cpu",
    blend=0.35,
    steps=36,
    previous_context=None,
    next_context=None,
    fps=30.0,
):
    from training.transition_diffusion import sample_transition_diffusion as implementation

    layout = normalize_rot6d_layout(
        bundle["rot6d_layout"] if bundle is not None else ROT6D_LAYOUT_PYTORCH3D_ROW
    )

    def to_native(value):
        if value is None:
            return None
        return convert_motion_rot6d_layout_np(
            value,
            CANONICAL_ROT6D_LAYOUT,
            layout,
        )

    generated, metadata = implementation(
        bundle,
        to_native(start_frame),
        to_native(end_frame),
        length,
        music_query,
        rough=to_native(rough),
        device=device,
        blend=blend,
        steps=steps,
        previous_context=to_native(previous_context),
        next_context=to_native(next_context),
        fps=fps,
    )
    output = convert_motion_rot6d_layout_np(
        generated,
        layout,
        CANONICAL_ROT6D_LAYOUT,
    )
    metadata = dict(metadata)
    metadata["checkpoint_rot6d_layout"] = layout
    metadata["output_rot6d_layout"] = CANONICAL_ROT6D_LAYOUT
    return output, metadata
