itk_module_test()

set(HalideFiltersTests
  itkHalideDiscreteGaussianImageFilterTest.cxx
  )

CreateTestDriver(HalideFilters "${HalideFilters-Test_LIBRARIES}" "${HalideFiltersTests}")

itk_add_test(NAME itkHalideDiscreteGaussianImageFilterTest
  COMMAND HalideFiltersTestDriver
    --compare
    DATA{Baseline/itkHalideDiscreteGaussianImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkHalideDiscreteGaussianImageFilterTestOutput.mha
  itkHalideDiscreteGaussianImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkHalideDiscreteGaussianImageFilterTestOutput.mha
  )
