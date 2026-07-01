"""Device and dtype selection utilities for CRA experiments."""

from typing import Dict, List, Optional


def get_device(device_str: str = "auto"):
    """Resolve a device string to a ``torch.device``.

    Parameters
    ----------
    device_str:
        ``"auto"`` selects CUDA if available, then MPS, then CPU.
        Any other value is passed directly to ``torch.device``.

    Returns
    -------
    torch.device
    """
    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device_str)


def get_dtype(dtype_str: str = "float32"):
    """Map a string dtype name to the corresponding ``torch.dtype``.

    Supported names: ``"float32"`` / ``"fp32"``, ``"float16"`` / ``"fp16"``,
    ``"bfloat16"`` / ``"bf16"``.

    Returns
    -------
    torch.dtype
    """
    import torch

    _MAP = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    key = dtype_str.lower().strip()
    if key not in _MAP:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. "
            f"Supported values: {list(_MAP.keys())}"
        )
    return _MAP[key]


def get_gpu_info() -> List[Dict[str, object]]:
    """Return a list of dicts describing each visible CUDA GPU.

    Returns an empty list when CUDA is unavailable.

    Each dict contains:
    - ``index`` (int)
    - ``name`` (str)
    - ``total_memory_gb`` (float)
    - ``free_memory_gb`` (float)
    - ``compute_capability`` (str)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 2),
                    "free_memory_gb": round(free / 1e9, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        return info

    except Exception:
        return []
