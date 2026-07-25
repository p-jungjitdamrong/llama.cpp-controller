"""Minimal GGUF header reader.

Reads just enough of the metadata key/value block to describe a model in the UI
(architecture, training context, layer count, quantisation) without pulling in
any dependency. Large arrays such as the tokenizer vocabulary are skipped
rather than materialised.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"

# ggml_type ids used by the general.file_type metadata key
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}
_STRING = 8
_ARRAY = 9

# Only these keys are worth keeping; everything else is skipped.
_WANTED_SUFFIXES = (
    "architecture", "name", "basename", "size_label", "file_type",
    "quantization_version", "parameter_count", "context_length",
    "block_count", "embedding_length", "attention.head_count",
    "attention.head_count_kv", "rope.freq_base", "expert_count",
)


def _read(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("unexpected end of GGUF header")
    return data


def _read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _read_str(f: BinaryIO) -> str:
    return _read(f, _read_u64(f)).decode("utf-8", errors="replace")


def _skip_value(f: BinaryIO, vtype: int) -> None:
    if vtype in _SCALARS:
        f.seek(_SCALARS[vtype][1], 1)
    elif vtype == _STRING:
        f.seek(_read_u64(f), 1)
    elif vtype == _ARRAY:
        itype = struct.unpack("<I", _read(f, 4))[0]
        count = _read_u64(f)
        if itype in _SCALARS:
            f.seek(_SCALARS[itype][1] * count, 1)
        elif itype == _STRING:
            for _ in range(count):
                f.seek(_read_u64(f), 1)
        else:
            raise ValueError(f"unsupported array element type {itype}")
    else:
        raise ValueError(f"unsupported value type {vtype}")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        fmt, size = _SCALARS[vtype]
        return struct.unpack(fmt, _read(f, size))[0]
    if vtype == _STRING:
        return _read_str(f)
    _skip_value(f, vtype)  # arrays are never kept
    return None


def read_metadata(path: Path) -> dict[str, Any]:
    """Return selected GGUF metadata, or {"error": ...} if the file is unreadable."""
    try:
        with path.open("rb") as f:
            if _read(f, 4) != MAGIC:
                return {"error": "not a GGUF file"}
            version = struct.unpack("<I", _read(f, 4))[0]
            tensor_count = _read_u64(f)
            kv_count = _read_u64(f)

            kv: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _read_str(f)
                vtype = struct.unpack("<I", _read(f, 4))[0]
                if key.endswith(_WANTED_SUFFIXES):
                    value = _read_value(f, vtype)
                    if value is not None:
                        kv[key] = value
                else:
                    _skip_value(f, vtype)
    except (OSError, ValueError, struct.error) as exc:
        return {"error": str(exc)}

    arch = kv.get("general.architecture", "")
    out: dict[str, Any] = {
        "gguf_version": version,
        "tensor_count": tensor_count,
        "architecture": arch,
        "meta_name": kv.get("general.name", ""),
        "size_label": kv.get("general.size_label", ""),
        "param_count": kv.get("general.parameter_count"),
        "n_ctx_train": kv.get(f"{arch}.context_length"),
        "n_layer": kv.get(f"{arch}.block_count"),
        "n_embd": kv.get(f"{arch}.embedding_length"),
        "n_head": kv.get(f"{arch}.attention.head_count"),
        "n_head_kv": kv.get(f"{arch}.attention.head_count_kv"),
        "n_expert": kv.get(f"{arch}.expert_count"),
    }
    ftype = kv.get("general.file_type")
    if ftype is not None:
        out["quant"] = FILE_TYPES.get(ftype, f"type {ftype}")
    return {k: v for k, v in out.items() if v not in (None, "")}
