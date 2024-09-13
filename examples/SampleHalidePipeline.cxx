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

#include "itkHalideDiscreteGaussianImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkHalideGPUDiscreteGaussianImageFilter.h"
#include "itkGPUDiscreteGaussianImageFilter.h"
#include "itkAdditiveGaussianNoiseImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImage.h"
#include "itkGPUImage.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#define BENCH(label, samples, block)                                                      \
  do                                                                                      \
  {                                                                                       \
    auto start = std::chrono::high_resolution_clock::now();                               \
    for (size_t __i = 0; __i < samples; __i++)                                            \
    {                                                                                     \
      block;                                                                              \
    }                                                                                     \
    auto end = std::chrono::high_resolution_clock::now();                                 \
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); \
    std::cout << #label << " " << (ms / samples) << "ms" << std::endl;                    \
  } while (0)

using ImageType = itk::Image<float, 3>;
using NoiseFilter = itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType>;
using GPUImageType = itk::GPUImage<float, 3>;
using CastToGPUImage = itk::CastImageFilter<ImageType, GPUImageType>;

using CPUBlur = itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
using HalideBlur = itk::HalideDiscreteGaussianImageFilter<ImageType, ImageType>;

using GPUBlur = itk::GPUDiscreteGaussianImageFilter<GPUImageType, GPUImageType>;
using HalideGPUBlur = itk::HalideGPUDiscreteGaussianImageFilter<ImageType, ImageType>;

int
main(int argc, char * argv[])
{
  ImageType::IndexValueType SIZE = 300;
  float                     VARIANCE = 90;

  ImageType::IndexType index;
  index.Fill(0);
  ImageType::SizeType size;
  size.Fill(SIZE);
  ImageType::RegionType region;
  region.SetIndex(index);
  region.SetSize(size);

  ImageType::Pointer image = ImageType::New();
  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0.0f);

  NoiseFilter::Pointer noise = NoiseFilter::New();
  noise->SetInput(image);
  noise->SetMean(0);
  noise->SetStandardDeviation(2.0);
  noise->Update();

  image = noise->GetOutput();

  BENCH(cpu_blur, 1, {
    CPUBlur::Pointer filter = CPUBlur::New();
    filter->SetInput(image);
    filter->SetVariance(VARIANCE);
    filter->Update();
  });

  BENCH(halide_cpu_blur, 1, {
    HalideBlur::Pointer filter = HalideBlur::New();
    filter->SetInput(image);
    filter->SetVariance(VARIANCE);
    filter->Update();
  });

  BENCH(gpu_blur, 1, {
    CastToGPUImage::Pointer cast = CastToGPUImage::New();
    cast->SetInput(image);
    GPUBlur::Pointer      filter = GPUBlur::New();
    GPUImageType::Pointer gpu_image = cast->GetOutput();
    cast->Update();
    filter->SetInput(gpu_image);
    filter->SetVariance(VARIANCE);
    filter->Update();
    filter->GetOutput()->UpdateBuffers();
  });

  BENCH(halide_gpu_blur, 1, {
    HalideGPUBlur::Pointer filter = HalideGPUBlur::New();
    filter->SetInput(image);
    filter->SetVariance(VARIANCE);
    filter->Update();
  });

  return EXIT_SUCCESS;
}
