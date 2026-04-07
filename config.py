"""Model registry and shared configuration."""

from __future__ import annotations

import os

import yaml
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = "results"
DASHBOARD_DATA_DIR = "dashboard/data"

# Load model registry from YAML
_config_path = os.path.join(os.path.dirname(__file__), "models.yaml")
with open(_config_path) as _f:
    _raw = yaml.safe_load(_f)

MODEL_REGISTRY: dict[str, dict] = _raw["models"]
DEFAULT_SYSTEM_PROMPT: str = _raw.get("default_system_prompt", "")

API_KEYS = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "xai": os.getenv("XAI_API_KEY", ""),
    "google": os.getenv("GOOGLE_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
}
