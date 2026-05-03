from __future__ import annotations

from pathlib import Path

from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CARCASSONNE_GAME = ROOT / "open_spiel" / "games" / "carcassonne" / "game"
LOCAL_PYBIND11_INCLUDE = ROOT / "pybind11" / "include"


def pybind11_include() -> str:
    try:
        import pybind11
    except ImportError as exc:  # pragma: no cover - setup-time dependency guard
        if LOCAL_PYBIND11_INCLUDE.exists():
            return path(LOCAL_PYBIND11_INCLUDE)
        raise SystemExit(
            "pybind11 is required. Install play requirements first: python -m pip install -r play/requirements.txt"
        ) from exc

    get_include = getattr(pybind11, "get_include", None)
    if get_include is not None:
        return get_include()
    if LOCAL_PYBIND11_INCLUDE.exists():
        return path(LOCAL_PYBIND11_INCLUDE)
    raise SystemExit("Could not locate pybind11 headers.")


def path(value: Path) -> str:
    return str(value)


extension = Extension(
    "play._carcassonne_cpp",
    sources=[
        path(HERE / "_carcassonne_cpp.cpp"),
        path(CARCASSONNE_GAME / "game.cpp"),
        path(CARCASSONNE_GAME / "BoardModule.cpp"),
        path(CARCASSONNE_GAME / "DeckModule.cpp"),
        path(CARCASSONNE_GAME / "Feature.cpp"),
        path(CARCASSONNE_GAME / "FeatureModule.cpp"),
        path(CARCASSONNE_GAME / "FrontierModule.cpp"),
        path(CARCASSONNE_GAME / "MonasteryModule.cpp"),
    ],
    include_dirs=[
        pybind11_include(),
        path(ROOT),
        path(ROOT / "open_spiel"),
        path(CARCASSONNE_GAME),
    ],
    language="c++",
    extra_compile_args=["-std=c++17"],
)


setup(
    name="carcassonne-play",
    version="0.1.0",
    description="Flet UI bridge for the local Carcassonne engine.",
    ext_modules=[extension],
)
