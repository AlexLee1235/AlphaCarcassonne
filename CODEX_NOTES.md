# Codex Notes

## OpenSpiel build/configure rule

- Do not run a bare configure such as `cmake -S open_spiel -B build`.
- The project build settings are controlled by `open_spiel/scripts/global_variables.sh`.
- For normal full rebuilds, use the repo script:

```bash
./open_spiel/scripts/build_and_run_tests.sh
```

- If a new CMake target was added and the existing `build/` directory must be regenerated, mirror the script instead of inventing a new configure command:

```bash
source open_spiel/scripts/global_variables.sh
source venv/bin/activate
cd build
cmake -DPython3_EXECUTABLE=python \
      -DCMAKE_CXX_COMPILER=${CXX} \
      -DCMAKE_PREFIX_PATH=${LIBCXXWRAP_JULIA_DIR} \
      -DBUILD_TYPE=Testing \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      ../open_spiel
```

- After configure, build only the requested target:

```bash
cmake --build build --target <target> -j2
```
