ITKHalideFilters
=================================

.. image:: https://github.com/KitwareMedical/ITKHalideFilters/actions/workflows/build-test-package.yml/badge.svg
    :target: https://github.com/KitwareMedical/ITKHalideFilters/actions/workflows/build-test-package.yml
    :alt: Build Status

.. image:: https://img.shields.io/pypi/v/itk-halidefilters.svg
    :target: https://pypi.python.org/pypi/itk-halidefilters
    :alt: PyPI Version

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://github.com/KitwareMedical/ITKHalideFilters/blob/main/LICENSE
    :alt: License

Overview
--------

Halide implementation of common filters.

Experimental integration of Halide runtime into ITK, with Halide implementations of common filters targeting Threads, SIMD, and GPGPU.

Building
--------

Build ITK with flag `ITK_USE_GPU`. 

.. code-block:: bash

  git clone https://github.com/InsightSoftwareConsortium/ITK.git -b release-5.4
  cmake -S ITK -B ITK-build -DITK_USE_GPU=ON
  cmake --build ITK-build --parallel

Build ITKHalideFilters against that ITK build. 

.. code-block:: bash

  git clone https://github.com/KitwareMedical/ITKHalideFilters.git
  cmake -S ITKHalideFilters -B ITKHalideFilters-build -DITK_DIR=ITK-build
  cmake --build ITKHalideFilters-build --parallel

The ITKHalideFilters build system will automatically download an appropriate version of Halide. Options may be given with `-D`, or set in `ccmake` or `cmake-gui`.

- ``-DModule_HalideFilters_TEST_GPU=ON`` (default OFF) will enable tests for GPU filters. This is disabled by default as the CI server does not have GPU provisioning. Halide guarantees that all schedules of the same algorithm will produce the same output, so CI testing the CPU schedule **should** be sufficient to check correctness. Recommend setting this ON for local builds with CUDA availability.

- ``-DModule_HalideFilters_USE_AUTOSCHEDULER=ON`` (default OFF) will invoke appropriate autoschedulers for the Halide filters. This significantly increases build time, but may improve runtime performance on specific hardware.

