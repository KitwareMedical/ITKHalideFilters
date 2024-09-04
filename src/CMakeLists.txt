find_package(Halide REQUIRED shared)

add_executable(itkHalideGenerators generators.cpp)
target_link_libraries(itkHalideGenerators PRIVATE Halide::Generator)

add_halide_library(itkHalideDiscreteGaussianImpl
  FROM itkHalideGenerators
  HEADER itkHalideDiscreteGaussianImpl_h
  # PARAMS input.type=float32 output.type=float32
  FEATURES cuda # debug
  )

set(HalideFilters_SRCS
  ${itkHalideDiscreteGaussianImpl_h}
  )

itk_module_add_library(HalideFilters ${HalideFilters_SRCS})
target_include_directories(HalideFilters PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(HalideFilters PUBLIC itkHalideDiscreteGaussianImpl)
