itk_module_test()

set(HalideFiltersTests
  itkHalideDiscreteGaussianImageFilterTest.cxx
  itkHalideGPUDiscreteGaussianImageFilterTest.cxx
  )

CreateTestDriver(HalideFilters "${HalideFilters-Test_LIBRARIES}" "${HalideFiltersTests}")

# NOTE: Input data is CTChest Slicer sample data: anisotropic image with blur variance 9mm.
# Reference output generated with itk::DiscreteGaussianImageFilter (via Slicer Simple Filters).
itk_add_test(NAME itkHalideDiscreteGaussianImageFilterTest
  COMMAND
  HalideFiltersTestDriver
  --compare DATA{CTChest/ReferenceOutput.mha} ${ITK_TEST_OUTPUT_DIR}/Output.mha
  itkHalideDiscreteGaussianImageFilterTest
  DATA{CTChest/Input.mha}
  ${ITK_TEST_OUTPUT_DIR}/Output.mha
  9
  )

if(Module_HalideFilters_TEST_GPU)
  itk_add_test(NAME itkHalideGPUDiscreteGaussianImageFilterTest
    COMMAND
    HalideFiltersTestDriver
    --compare DATA{CTChest/ReferenceOutput.mha} ${ITK_TEST_OUTPUT_DIR}/Output.mha
    itkHalideGPUDiscreteGaussianImageFilterTest
    DATA{CTChest/Input.mha}
    ${ITK_TEST_OUTPUT_DIR}/Output.mha
    9
    )
endif()