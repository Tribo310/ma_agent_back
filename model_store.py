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
        if isinstance(model, dict):
            tensor_values = sum(1 for v in model.values() if torch.is_tensor(v))
            wrapper_keys = {"state_dict", "model_state_dict", "model"} & model.keys()
            if wrapper_keys:
                raise ValueError(
                    f"Uploaded a checkpoint dict (keys: {list(model.keys())[:5]}…), not a model. "
                    "Re-save the trained module itself: "
                    "`torch.jit.save(torch.jit.script(model), 'model.pt')`."
                )
            if tensor_values and tensor_values == len(model):
                raise ValueError(
                    f"Uploaded a state_dict ({tensor_values} tensors), not a model. "
                    "A state_dict has no architecture, so the server cannot run it. "
                    "Re-save the module: `torch.jit.save(torch.jit.script(model), 'model.pt')`."
                )
        raise ValueError(
            f"Loaded object is not callable (got {type(model).__name__}) — "
            "expected nn.Module or TorchScript ScriptModule."
        )

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
