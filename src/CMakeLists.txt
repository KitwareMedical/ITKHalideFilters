find_package(Halide REQUIRED shared)

add_executable(itkHalideGenerators generators.cpp)
target_link_libraries(itkHalideGenerators PRIVATE Halide::Generator)

if(Module_HalideFilters_USE_AUTOSCHEDULER)
  add_halide_library(itkHalideSeparableConvolutionImpl
    FROM itkHalideGenerators
    GENERATOR itkHalideSeparableConvolutionImpl
    HEADER itkHalideSeparableConvolutionImpl_h
    SCHEDULE itkHalideSeparableConvolutionSchedule
    AUTOSCHEDULER Halide::Adams2019
    )

  add_halide_library(itkHalideGPUSeparableConvolutionImpl
    FROM itkHalideGenerators
    GENERATOR itkHalideSeparableConvolutionImpl
    HEADER itkGPUHalideSeparableConvolutionImpl_h
    SCHEDULE itkHalideGPUSeparableConvolutionSchedule
    FEATURES cuda
    AUTOSCHEDULER Halide::Anderson2021
    )
else()
  add_halide_library(itkHalideSeparableConvolutionImpl
    FROM itkHalideGenerators
    GENERATOR itkHalideSeparableConvolutionImpl
    HEADER itkHalideSeparableConvolutionImpl_h
    PARAMS use_gpu=false
    )

  add_halide_library(itkHalideGPUSeparableConvolutionImpl
    FROM itkHalideGenerators
    GENERATOR itkHalideSeparableConvolutionImpl
    HEADER itkGPUHalideSeparableConvolutionImpl_h
    FEATURES cuda
    PARAMS use_gpu=true
    )
endif()

set(HalideFilters_SRCS
  ${itkHalideGPUSeparableConvolutionImpl_h}
  ${itkHalideSeparableConvolutionImpl_h}
  )

itk_module_add_library(HalideFilters ${HalideFilters_SRCS})
target_include_directories(HalideFilters PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(HalideFilters PUBLIC itkHalideSeparableConvolutionImpl itkHalideGPUSeparableConvolutionImpl)
set_target_properties(HalideFilters PROPERTIES LINKER_LANGUAGE CXX)
