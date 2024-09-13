/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkHalideGPUDiscreteGaussianImageFilter_hxx
#define itkHalideGPUDiscreteGaussianImageFilter_hxx

#include "itkHalideGPUDiscreteGaussianImageFilter.h"

#include "itkHalideGPUSeparableConvolutionImpl.h"

#include "itkGaussianOperator.h"

#include <Halide.h>
#include <HalideBuffer.h>
#include <iomanip>

namespace itk
{

template <typename TInputImage, typename TOutputImage>
HalideGPUDiscreteGaussianImageFilter<TInputImage, TOutputImage>::HalideGPUDiscreteGaussianImageFilter()
{
  this->DynamicMultiThreadingOff();
  this->ThreaderUpdateProgressOff();
}


template <typename TInputImage, typename TOutputImage>
void
HalideGPUDiscreteGaussianImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


template <typename TInputImage, typename TOutputImage>
void
HalideGPUDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  const InputImageType *               input = this->GetInput();
  typename InputImageType::RegionType  inputRegion = input->GetBufferedRegion();
  typename InputImageType::SizeType    inputSize = inputRegion.GetSize();
  typename InputImageType::SpacingType inputSpacing = input->GetSpacing();

  std::vector<Halide::Runtime::Buffer<float, 1>> kernel_buffers{};

  // compute kernel coefficients with itk::GaussianOperator to match behavior with itk::DiscreteGaussianImageFilter
  for (int dim = 0; dim < InputImageDimension; ++dim)
  {
    GaussianOperator<float, 1> oper{};
    oper.SetMaximumError(m_MaximumError);
    oper.SetMaximumKernelWidth(m_MaximumKernelWidth);

    float variance = m_Variance;
    if (m_UseImageSpacing)
    {
      variance /= inputSpacing[dim];
    }
    oper.SetVariance(variance);

    oper.CreateDirectional();

    Halide::Runtime::Buffer<float, 1> & buf = kernel_buffers.emplace_back(static_cast<int>(oper.GetSize(0)));
    buf.set_min(-static_cast<int>(oper.GetRadius(0)));
    std::copy(oper.Begin(), oper.End(), buf.begin());
    buf.set_host_dirty();
  }

  OutputImageType * output = this->GetOutput();
  output->SetRegions(inputRegion);
  output->Allocate();

  std::vector<int> sizes(3, 1);
  std::copy(inputSize.begin(), inputSize.end(), sizes.begin());

  Halide::Runtime::Buffer<const InputPixelType> inputBuffer(input->GetBufferPointer(), sizes);
  Halide::Runtime::Buffer<OutputPixelType>      outputBuffer(output->GetBufferPointer(), sizes);

  inputBuffer.set_host_dirty();
  itkHalideGPUSeparableConvolutionImpl(inputBuffer, kernel_buffers[0], kernel_buffers[1], kernel_buffers[2], outputBuffer);
  outputBuffer.copy_to_host();
}

} // end namespace itk

#endif // itkHalideGPUDiscreteGaussianImageFilter_hxx
