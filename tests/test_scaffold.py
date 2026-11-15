"""Initial scaffold verification test."""

from pathlib import Path

import yaml


def test_directory_structure():
    """Verify that all core directories and config files exist."""
    root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "configs",
        "src/hotel_forecast/data",
        "src/hotel_forecast/features",
        "src/hotel_forecast/models",
        "src/hotel_forecast/api",
        "app",
        "notebooks",
        "tests",
        "reports",
        "reports/figures",
        "docs",
    ]
    for d in expected_dirs:
        assert (root / d).is_dir(), f"Missing required directory: {d}"


def test_config_validity():
    """Verify that config.yaml parses and contains core project keys."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "config.yaml"
    assert config_path.is_file(), "config.yaml not found"

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["project"]["seed"] == 42
    assert "Standard" in cfg["hotel"]["categories"]
    assert cfg["hotel"]["total_capacity"] == 120
    assert cfg["forecast"]["horizon_days"] == 14
    assert cfg["splits"]["purge_gap_days"] == 14

