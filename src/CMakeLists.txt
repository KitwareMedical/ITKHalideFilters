find_package(Halide REQUIRED shared)

add_executable(my_generators generators.cpp)
target_link_libraries(my_generators PRIVATE Halide::Generator)
add_halide_library(my_first_generator FROM my_generators HEADER my_first_generator_h)
add_halide_library(cast_image_generator FROM my_generators HEADER cast_image_generator_h PARAMS output.type=uint32)

set(HalideFilters_SRCS
  itkMinimalStandardRandomVariateGenerator.cxx
  ${my_first_generator_h}
  ${cast_image_generator_h}
  )

itk_module_add_library(HalideFilters ${HalideFilters_SRCS})
target_include_directories(HalideFilters PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(HalideFilters PRIVATE Halide::RunGenMain cast_image_generator)
