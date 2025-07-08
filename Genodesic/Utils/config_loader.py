import yaml
from pathlib import Path
from typing import Dict, Any

def _deep_merge(src: dict, dst: dict) -> dict:
    """Recursively merge *src* into *dst* (src wins)."""
    for k, v in src.items():
        if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
            _deep_merge(v, dst[k])
        else:
            dst[k] = v
    return dst

def load_config(
    default_config_path: str,
    overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and merges it with an optional override dictionary.

    Args:
        default_config_path (str): The path to the base YAML configuration file.
        overrides (dict, optional): A dictionary of parameters to override.

    Returns:
        A single, fully-resolved configuration dictionary.
    """
    if not Path(default_config_path).exists():
        raise FileNotFoundError(f"Default config not found at: {default_config_path}")
    
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        config = _deep_merge(overrides, config)
        
    return config