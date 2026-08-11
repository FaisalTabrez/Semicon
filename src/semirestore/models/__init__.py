"""Restoration model definitions."""

from .bicubic import BicubicRestorer
from .edsr_lite import EDSRLite


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
    raise ValueError(f"Unsupported model name: {name}")


__all__ = ["BicubicRestorer", "EDSRLite", "create_model"]
