"""Restoration model definitions."""

from .bicubic import BicubicRestorer
from .edsr_lite import EDSRLite
from .naf_sr import NAFSR


def create_model(name: str, config: dict[str, object] | None = None):
    """Construct a restoration model from self-describing checkpoint metadata."""

    resolved = config or {}
    if name == "bicubic":
        if resolved:
            raise ValueError("The bicubic model does not accept configuration values")
        return BicubicRestorer()
    if name == "edsr_lite":
        allowed = {"width", "num_blocks", "residual_scale"}
        unknown = resolved.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown EDSR-lite configuration field(s): {sorted(unknown)}")
        return EDSRLite(**resolved)
    if name == "naf_sr":
        allowed = {
            "width",
            "encoder_blocks",
            "middle_blocks",
            "decoder_blocks",
            "dropout",
        }
        unknown = resolved.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown NAF-SR configuration field(s): {sorted(unknown)}")
        return NAFSR(**resolved)
    raise ValueError(f"Unsupported model name: {name}")


__all__ = ["BicubicRestorer", "EDSRLite", "NAFSR", "create_model"]
