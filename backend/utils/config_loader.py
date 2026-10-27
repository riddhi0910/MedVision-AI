"""
src/utils/config_loader.py
==========================
Reads config.yaml and .env, and exposes a single `get_config()` function
that returns a plain Python dictionary the rest of the code can use.

Why YAML?
  - Human-readable key=value pairs
  - Supports nested structure (sections for dataset, model, training …)
  - Much better than hard-coding numbers scattered across files

Why .env?
  - Keeps personal paths (like your dataset location) out of the shared
    config.yaml so you never accidentally commit them.
"""

import os
import yaml                     # reads .yaml files
from pathlib import Path        # handles file paths cleanly on any OS
from dotenv import load_dotenv  # reads .env into os.environ


def get_config(config_path: str | None = None) -> dict:
    """
    Load and return the project configuration as a dictionary.

    Steps:
    1. Find the project root (the folder that contains configs/).
    2. Load environment variables from .env (if it exists).
    3. Read configs/config.yaml.
    4. Override DATASET_ROOT if the env variable is set.

    Parameters
    ----------
    config_path : str or None
        Path to an alternative config file.  Leave None to use the default.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """

    # --- 1. Locate the project root -----------------------------------
    this_file = Path(__file__).resolve()
    # Search upwards for the directory containing configs/config.yaml
    project_root = this_file.parents[2]
    for p in this_file.parents:
        if (p / "configs" / "config.yaml").exists():
            project_root = p
            break

    # --- 2. Load .env -------------------------------------------------
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[Config] Loaded environment from: {env_file}")
    else:
        print("[Config] No .env file found. Using system environment variables.")

    # --- 3. Read config.yaml ------------------------------------------
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)   # safe_load prevents arbitrary code execution

    # --- 4. Apply .env overrides --------------------------------------
    dataset_root_env = os.getenv("DATASET_ROOT")
    if dataset_root_env:
        cfg["dataset"]["root"] = dataset_root_env
        print(f"[Config] Dataset root overridden by .env: {dataset_root_env}")

    device_env = os.getenv("DEVICE")
    if device_env:
        cfg["device_override"] = device_env

    seed_env = os.getenv("RANDOM_SEED")
    if seed_env:
        cfg["dataset"]["random_seed"] = int(seed_env)

    # Store project root in config for convenience
    cfg["project_root"] = str(project_root)

    return cfg


if __name__ == "__main__":
    # Quick test: run this file directly to check config loading works.
    # Command: python src/utils/config_loader.py
    config = get_config()
    print("\n=== Loaded Configuration ===")
    import json
    print(json.dumps(config, indent=2, default=str))
