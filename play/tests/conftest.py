from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _game_log_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Finished games are saved; keep the ones tests play out of the real game directory."""
    directory = tmp_path / "games"
    monkeypatch.setenv("CARCASSONNE_GAME_LOG_DIR", str(directory))
    return directory
