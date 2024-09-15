cmake_minimum_required(VERSION 3.16.3)
project(HalideFiltersExamples)

set(ExampleSpecificComponents
  HalideFilters
  ITKGPUSmoothing
  ITKImageNoise
  )

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED COMPONENTS ITKImageIO ITKTransformIO ${ExampleSpecificComponents})
else()
  # When being built as part of ITK, ITKImageIO and ITKTransformIO
  # lists of modules are not yet ready, causing a configure error
  find_package(ITK REQUIRED COMPONENTS ${ExampleSpecificComponents})
endif()
include(${ITK_USE_FILE})

add_executable(ResolutionBenchmark ResolutionBenchmark.cxx)
target_link_libraries(ResolutionBenchmark ${ITK_LIBRARIES})

add_executable(SigmaBenchmark SigmaBenchmark.cxx)
target_link_libraries(SigmaBenchmark ${ITK_LIBRARIES})
