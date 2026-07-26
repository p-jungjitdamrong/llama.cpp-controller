"""Runtime configuration for the controller, persisted as JSON next to the app."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("LLAMACTL_CONFIG", APP_DIR / "config.json"))


@dataclass
class LaunchParams:
    """Flags handed to llama-server when a model is started."""

    # "single" runs one model; "router" lets llama-server host a directory of
    # models on one port and load them on demand
    mode: str = "single"
    # 0.0.0.0 so the model API is reachable from other machines, not just localhost
    host: str = "0.0.0.0"
    port: int = 8090
    # router-only
    models_dir: str = ""
    models_max: int = 2
    models_autoload: bool = True
    n_gpu_layers: int = 99
    ctx_size: int = 4096
    threads: int = 0  # 0 = let llama.cpp decide
    parallel: int = 1
    batch_size: int = 0  # 0 = llama.cpp default
    flash_attn: str = "auto"  # on | off | auto
    mlock: bool = False
    no_mmap: bool = False
    jinja: bool = True
    extra_args: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "LaunchParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class Config:
    llama_server_bin: str = "~/llama.cpp/build/bin/llama-server"
    model_dirs: list[str] = field(
        default_factory=lambda: [
            "~/models",
            "~/llama.cpp/models",
            # downloads made by `llama-server -hf …` and by huggingface-cli
            "~/.cache/llama.cpp",
            "~/.cache/huggingface/hub",
            "~",
        ]
    )
    controller_host: str = "0.0.0.0"
    controller_port: int = 8080
    history_size: int = 300  # metric samples kept in memory (~5 min at 1 Hz)
    log_buffer_lines: int = 5000
    defaults: LaunchParams = field(default_factory=LaunchParams)
    # model path -> saved launch params
    presets: dict[str, dict] = field(default_factory=dict)
    # servers to bring up when the controller starts, in order:
    # [{"model_path": "...", "params": {...}}, …]
    autostart: list[dict] = field(default_factory=list)
    # last model started, restored into the UI on reload
    last_model: str = ""

    @property
    def server_bin(self) -> Path:
        return Path(self.llama_server_bin).expanduser()

    def resolved_model_dirs(self) -> list[Path]:
        seen: list[Path] = []
        for raw in self.model_dirs:
            p = Path(raw).expanduser()
            if p not in seen:
                seen.append(p)
        return seen

    def download_dir(self) -> Path:
        """Where Hub downloads land: the first configured directory, or ~/models."""
        for raw in self.model_dirs:
            path = Path(raw).expanduser()
            if path.name not in ("hub", "llama.cpp") and path != Path.home():
                return path
        return Path.home() / "models"

    def params_for(self, model_path: str) -> LaunchParams:
        saved = self.presets.get(model_path)
        return LaunchParams.from_dict(saved) if saved else LaunchParams(**asdict(self.defaults))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            cfg = cls()
            cfg.save()
            return cfg
        raw = json.loads(CONFIG_PATH.read_text())
        defaults = LaunchParams.from_dict(raw.pop("defaults", {}))
        known = {f for f in cls.__dataclass_fields__} - {"defaults"}
        cfg = cls(defaults=defaults, **{k: v for k, v in raw.items() if k in known})
        return cfg

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
