if (OPEN_SPIEL_BUILDING_WHEEL)
  # When building a wheel, need to use this. See:
  # https://github.com/joerick/cibuildwheel/issues/639#issuecomment-817872369
  message(NOTICE "Building Wheel Detected. Finding Python Interpeter Development.Module")
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  unset(OPEN_SPIEL_PYTHONPATH)
  if(DEFINED ENV{PYTHONPATH})
    set(OPEN_SPIEL_PYTHONPATH $ENV{PYTHONPATH})
  endif()
else()
  message(NOTICE "Not building wheel. Finding Python normally...")
  find_package(Python3 COMPONENTS Interpreter Development)
  set(OPEN_SPIEL_PYTHONPATH ${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;)
endif()

message(NOTICE "Python executable: ${Python3_EXECUTABLE}")
message(NOTICE "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(NOTICE "Python library dirs: ${Python3_LIBRARY_DIRS}")
include_directories(${Python3_INCLUDE_DIRS})

if(Python3_VERSION VERSION_LESS "3.8.0")
  message(FATAL_ERROR
    "Python found ${Python3_VERSION} < 3.8.0")
endif()

include_directories(../pybind11_abseil ../../pybind11/include)
set(PYTHON_BINDINGS ${PYTHON_BINDINGS}
  pybind11/algorithms_trajectories.cc
  pybind11/algorithms_trajectories.h
  pybind11/bots.cc
  pybind11/bots.h
  pybind11/evaluation_elo.cc
  pybind11/evaluation_elo.h
  pybind11/evaluation_sco.cc
  pybind11/evaluation_sco.h
  pybind11/games_tic_tac_toe.cc
  pybind11/games_tic_tac_toe.h
  pybind11/observer.cc
  pybind11/observer.h
  pybind11/policy.cc
  pybind11/policy.h
  pybind11/pybind11.h
  pybind11/pyspiel.cc
  pybind11/python_games.cc
  pybind11/python_games.h
  pybind11/python_policy.cc
  pybind11/python_policy.h
  pybind11/utils.cc
  pybind11/utils.h
)

add_library(pyspiel MODULE ${PYTHON_BINDINGS} ${OPEN_SPIEL_OBJECTS})
target_link_directories(pyspiel PUBLIC ${Python3_LIBRARY_DIRS})

# Without this, the binary is called `libpyspiel.so`
set_target_properties(pyspiel PROPERTIES PREFIX "")
if (WIN32)
  # Python does not seem able to import the module without this change.
  set_target_properties(pyspiel PROPERTIES SUFFIX ".pyd")
endif()

if(NOT DEFINED ENV{PYBIN})
  set(PYBIN ${Python3_EXECUTABLE})
else()
  set(PYBIN $ENV{PYBIN})
endif()

set(PYTHON_TESTS
  tests/pyspiel_test.py
)

foreach(py_test_file IN LISTS PYTHON_TESTS)
  add_test(NAME python/${py_test_file} COMMAND ${PYBIN} ${CMAKE_CURRENT_SOURCE_DIR}/${py_test_file})
  set_property(TEST python/${py_test_file}
      PROPERTY ENVIRONMENT
      PYTHONPATH=${OPEN_SPIEL_PYTHONPATH}
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
endforeach(py_test_file)
