find_package(Halide REQUIRED shared)

add_executable(itkHalideGenerators generators.cpp)
target_link_libraries(itkHalideGenerators PRIVATE Halide::Generator)

add_halide_library(itkHalideSeparableConvolutionImpl
  FROM itkHalideGenerators
  HEADER itkHalideSeparableConvolutionImpl_h
  FEATURES cuda
  )

set(HalideFilters_SRCS
  ${itkHalideSeparableConvolutionImpl_h}
  )

itk_module_add_library(HalideFilters ${HalideFilters_SRCS})
target_include_directories(HalideFilters PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(HalideFilters PUBLIC itkHalideSeparableConvolutionImpl)
