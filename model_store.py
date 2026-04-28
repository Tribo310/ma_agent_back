"""
Module-level model registry.
Supports TorchScript (.pt saved via torch.jit.save) and full-model files.
"""
from __future__ import annotations

import io
from typing import Optional

import torch
import torch.nn as nn

_models: dict[str, nn.Module] = {}


def validate_and_store(team: str, model_bytes: bytes) -> None:
    """Load, validate output shape, then store the model.

    Raises ValueError with a human-readable message on any failure so the
    endpoint can forward it as HTTP 400.
    """
    buf = io.BytesIO(model_bytes)

    # Try TorchScript first (portable, no class definition needed)
    model: nn.Module
    try:
        model = torch.jit.load(buf, map_location="cpu")
    except Exception:
        buf.seek(0)
        try:
            model = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(
                f"Could not deserialise model: {exc}. "
                "Upload a TorchScript (.pt via torch.jit.save) or a pickled nn.Module."
            ) from exc

    if not callable(model):
        raise ValueError("Loaded object is not callable — expected nn.Module or ScriptModule.")

    model.eval()

    # Forward-pass validation
    dummy = torch.zeros(1, 13, 13, 5, dtype=torch.float32)
    try:
        with torch.no_grad():
            out = model(dummy)
    except Exception as exc:
        raise ValueError(
            f"Model forward pass failed with input shape (1,13,13,5): {exc}"
        ) from exc

    if out.shape[-1] != 21:
        raise ValueError(
            f"Model must output 21 logits (Discrete(21)); got shape {list(out.shape)}."
        )

    _models[team] = model


def get_model(team: str) -> Optional[nn.Module]:
    return _models.get(team)


def is_ready() -> bool:
    return "red" in _models and "blue" in _models


def red_ready() -> bool:
    return "red" in _models


def blue_ready() -> bool:
    return "blue" in _models
