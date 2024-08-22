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
#ifndef itkHalideCastImageFilter_hxx
#define itkHalideCastImageFilter_hxx

#include "itkHalideCastImageFilter.h"

#include "itkImportImageFilter.h"
#include "cast_image_generator.h"
#include <HalideBuffer.h>

namespace itk
{

template <typename TInputImage, typename TOutputImage>
HalideCastImageFilter<TInputImage, TOutputImage>::HalideCastImageFilter()
{
  this->DynamicMultiThreadingOff();
}


template <typename TInputImage, typename TOutputImage>
void
HalideCastImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


template <typename TInputImage, typename TOutputImage>
void
HalideCastImageFilter<TInputImage, TOutputImage>::GenerateData(const OutputRegionType & outputRegion)
{
  const InputImageType *              input = this->GetInput();
  typename InputImageType::RegionType inputRegion = input->GetLargestPosibleRegion();
  typename InputImageType::SizeType   inputSize = inputRegion.GetSize();

  std::vector<int> sizes;
  for (int i = 0; i < inputSize.size(); i++)
  {
    sizes.push_back(inputSize[i]);
  }

  Halide::Runtime::Buffer<const InputPixelType> inputBuffer(input->GetBufferPointer(), sizes);
  Halide::Runtime::Buffer<OutputPixelType>      outputBuffer(sizes);

  cast_image_generator(inputBuffer, outputBuffer);

  using ImportFilterType = itk::ImportImageFilter<OutputPixelType, OutputImageType::ImageDimension>;
  typename ImportFilterType::Pointer importFilter = ImportFilterType::New();

  importFilter->SetRegion(inputRegion);
  importFilter->SetOrigin(input->GetOrigin());
  importFilter->SetSpacing(input->GetSpacing());

  importFilter->SetImportPointer(outputBuffer.data(), outputBuffer.number_of_elements(), true);
  importFilter->Update();
  this->GraftOutput(importFilter->GetOutput());
}

template <typename TInputImage, typename TOutputImage>
void
HalideCastImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(const OutputRegionType & outputRegion)
{}

} // end namespace itk

#endif // itkHalideCastImageFilter_hxx
