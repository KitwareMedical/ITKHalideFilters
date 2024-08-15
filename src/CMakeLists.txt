set(HalideFilters_SRCS
  itkMinimalStandardRandomVariateGenerator.cxx
  )

find_package(Halide REQUIRED)
find_package(ZLIB REQUIRED)

itk_module_add_library(HalideFilters ${HalideFilters_SRCS})
target_link_libraries(HalideFilters PRIVATE Halide::Generator)
