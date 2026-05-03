try:
    from ui import run_app
except ImportError:  # pragma: no cover - package import fallback
    from .ui import run_app


if __name__ == "__main__":
    run_app()
