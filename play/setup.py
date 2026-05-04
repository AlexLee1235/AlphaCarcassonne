from __future__ import annotations

from pathlib import Path
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CARCASSONNE_GAME = ROOT / "open_spiel" / "games" / "carcassonne" / "game"
OPEN_SPIEL = ROOT / "open_spiel"
BUILD = ROOT / "build"
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


def absl_static_libs() -> list[str]:
    absl_root = BUILD / "abseil-cpp" / "absl"
    if not absl_root.exists():
        raise SystemExit(
            "OpenSpiel MCTS support needs the local OpenSpiel build's absl static libraries. "
            "Run the OpenSpiel CMake build first, then rerun: python play/setup.py build_ext --inplace"
        )
    libs = [path(value) for value in sorted(absl_root.rglob("libabsl_*.a"))]
    if not libs:
        raise SystemExit(
            "Could not find libabsl_*.a under build/abseil-cpp/absl. "
            "Run the OpenSpiel CMake build first, then rerun: python play/setup.py build_ext --inplace"
        )
    return libs


absl_libs = absl_static_libs()

carcassonne_sources = [
    CARCASSONNE_GAME / "game.cpp",
    CARCASSONNE_GAME / "BoardModule.cpp",
    CARCASSONNE_GAME / "DeckModule.cpp",
    CARCASSONNE_GAME / "Feature.cpp",
    CARCASSONNE_GAME / "FeatureModule.cpp",
    CARCASSONNE_GAME / "FrontierModule.cpp",
    CARCASSONNE_GAME / "MonasteryModule.cpp",
]


def require_dir(directory: Path, message: str) -> None:
    if not directory.exists():
        raise SystemExit(message)


def torch_link_args() -> list[str]:
    torch_root = OPEN_SPIEL / "libtorch" / "libtorch"
    torch_include = torch_root / "include"
    torch_lib = torch_root / "lib"
    require_dir(torch_include, "Missing libtorch headers under open_spiel/libtorch/libtorch/include.")
    require_dir(torch_lib, "Missing libtorch libraries under open_spiel/libtorch/libtorch/lib.")

    args = [
        f"-L{path(torch_lib)}",
        f"-Wl,-rpath,{path(torch_lib)}",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
    ]
    if (torch_lib / "libtorch_cuda.so").exists():
        args.extend(["-ltorch_cuda", "-lc10_cuda"])
    cuda_lib = Path("/usr/local/cuda/lib64")
    if cuda_lib.exists():
        args.extend([f"-L{path(cuda_lib)}", f"-Wl,-rpath,{path(cuda_lib)}", "-lcudart"])
        if (cuda_lib / "libnvToolsExt.so").exists():
            args.append("-lnvToolsExt")
    return args


def existing_objects(paths: list[Path], label: str) -> list[str]:
    missing = [path(value) for value in paths if not value.exists()]
    if missing:
        raise SystemExit(
            f"Missing {label} object files from the local OpenSpiel build. "
            "Run the OpenSpiel CMake build first. Missing: " + ", ".join(missing[:5])
        )
    return [path(value) for value in paths]


def open_spiel_core_objects() -> list[str]:
    core = BUILD / "CMakeFiles" / "open_spiel_core.dir"
    return existing_objects(
        [
            core / "action_view.cc.o",
            core / "canonical_game_strings.cc.o",
            core / "game_parameters.cc.o",
            core / "matrix_game.cc.o",
            core / "observer.cc.o",
            core / "policy.cc.o",
            core / "simultaneous_move_game.cc.o",
            core / "spiel.cc.o",
            core / "spiel_bots.cc.o",
            core / "spiel_utils.cc.o",
            core / "tensor_game.cc.o",
            core / "utils" / "status.cc.o",
            core / "utils" / "usage_logging.cc.o",
        ],
        "OpenSpiel core",
    )


def bot_algorithm_objects() -> list[str]:
    return existing_objects(
        [
            BUILD / "algorithms" / "CMakeFiles" / "algorithms.dir" / "mcts.cc.o",
            BUILD / "algorithms" / "alpha_zero_torch" / "CMakeFiles" / "alpha_zero_torch.dir" / "model.cc.o",
            BUILD / "algorithms" / "alpha_zero_torch" / "CMakeFiles" / "alpha_zero_torch.dir" / "vpevaluator.cc.o",
            BUILD / "algorithms" / "alpha_zero_torch" / "CMakeFiles" / "alpha_zero_torch.dir" / "vpnet.cc.o",
            BUILD / "utils" / "CMakeFiles" / "utils.dir" / "thread.cc.o",
        ],
        "OpenSpiel bot",
    )


class BuildWithBotCli(build_ext):
    def run(self) -> None:
        super().run()
        self.build_bot_cli()

    def build_bot_cli(self) -> None:
        output_dir = HERE / "bin"
        output_dir.mkdir(exist_ok=True)
        output = output_dir / "carcassonne_bot_cli"
        torch_root = OPEN_SPIEL / "libtorch" / "libtorch"
        sources = [
            HERE / "carcassonne_bot_cli.cpp",
            OPEN_SPIEL / "games" / "carcassonne" / "carcassonne.cc",
            *carcassonne_sources,
        ]
        command = [
            "g++",
            "-O2",
            "-std=c++17",
            "-DOPEN_SPIEL_BUILD_WITH_LIBNOP",
            "-DOPEN_SPIEL_BUILD_WITH_LIBTORCH",
            "-I" + path(ROOT),
            "-I" + path(OPEN_SPIEL),
            "-I" + path(OPEN_SPIEL / "abseil-cpp"),
            "-I" + path(OPEN_SPIEL / "json" / "include"),
            "-I" + path(OPEN_SPIEL / "libnop" / "libnop" / "include"),
            "-I" + path(torch_root / "include"),
            "-I" + path(torch_root / "include" / "torch" / "csrc" / "api" / "include"),
            "-I" + path(OPEN_SPIEL / "games" / "carcassonne"),
            "-I" + path(CARCASSONNE_GAME),
            *[path(source) for source in sources],
            *open_spiel_core_objects(),
            *bot_algorithm_objects(),
            "-o",
            path(output),
            "-Wl,--start-group",
            *absl_libs,
            "-Wl,--end-group",
            *torch_link_args(),
            "-pthread",
            "-ldl",
        ]
        subprocess.check_call(command, cwd=path(ROOT))


extension = Extension(
    "play._carcassonne_cpp",
    sources=[
        path(HERE / "_carcassonne_cpp.cpp"),
        *[path(source) for source in carcassonne_sources],
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
    cmdclass={"build_ext": BuildWithBotCli},
)
