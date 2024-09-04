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
#ifndef itkHalideDiscreteGaussianImageFilter_hxx
#define itkHalideDiscreteGaussianImageFilter_hxx

#include "itkHalideDiscreteGaussianImageFilter.h"

#include "itkHalideDiscreteGaussianImpl.h"

#include <HalideBuffer.h>

namespace itk
{

template <typename TInputImage, typename TOutputImage>
HalideDiscreteGaussianImageFilter<TInputImage, TOutputImage>::HalideDiscreteGaussianImageFilter()
{}


template <typename TInputImage, typename TOutputImage>
void
HalideDiscreteGaussianImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


template <typename TInputImage, typename TOutputImage>
void
HalideDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  const InputImageType *               input = this->GetInput();
  typename InputImageType::RegionType  inputRegion = input->GetBufferedRegion();
  typename InputImageType::SizeType    inputSize = inputRegion.GetSize();
  typename InputImageType::SpacingType inputSpacing = input->GetSpacing();

  const float sigma = GetSigma();
  const float sigma_x = sigma / inputSpacing[0];
  const float sigma_y = sigma / inputSpacing[1];
  const float sigma_z = sigma / inputSpacing[2];

  OutputImageType * output = this->GetOutput();
  output->SetRegions(inputRegion);
  output->Allocate();

  std::vector<int> sizes(3, 1);
  std::copy(inputSize.begin(), inputSize.end(), sizes.begin());

  Halide::Runtime::Buffer<const InputPixelType> inputBuffer(input->GetBufferPointer(), sizes);
  Halide::Runtime::Buffer<OutputPixelType>      outputBuffer(output->GetBufferPointer(), sizes);

  itkHalideDiscreteGaussianImpl(inputBuffer, sigma_x, sigma_y, sigma_z, outputBuffer);
}

} // end namespace itk

#endif // itkHalideDiscreteGaussianImageFilter_hxx
