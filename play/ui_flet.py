from __future__ import annotations

try:
    from ui.app import CarcassonneUI, main, run_app
except ImportError:  # pragma: no cover - package import fallback
    from .ui.app import CarcassonneUI, main, run_app

__all__ = ["CarcassonneUI", "main", "run_app"]
