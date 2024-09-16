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
#include "itkRecursiveGaussianImageFilter.h"
#include "itkHalideGPUDiscreteGaussianImageFilter.h"
#include "itkGPUDiscreteGaussianImageFilter.h"
#include "itkAdditiveGaussianNoiseImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImage.h"
#include "itkGPUImage.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

using ImageType = itk::Image<float, 3>;
using NoiseFilter = itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType>;
using GPUImageType = itk::GPUImage<float, 3>;
using CastToGPUImage = itk::CastImageFilter<ImageType, GPUImageType>;

using CPUBlur = itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
using HalideBlur = itk::HalideDiscreteGaussianImageFilter<ImageType, ImageType>;

using GPUBlur = itk::GPUDiscreteGaussianImageFilter<GPUImageType, GPUImageType>;
using HalideGPUBlur = itk::HalideGPUDiscreteGaussianImageFilter<ImageType, ImageType>;

using ns = std::chrono::duration<double, std::nano>;

ns
run_itk_cpu(ImageType * image, float sigma)
{
  using FilterType = itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetSigma(sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  filter->Update();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ns>(end - start);
}

ns
run_itk_rec(ImageType * image, float sigma)
{
  using FilterType = itk::RecursiveGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetSigma(sigma);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  filter->Update();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ns>(end - start);
}

ns
run_itk_gpu(ImageType * image, float sigma)
{
  using CastType = itk::CastImageFilter<ImageType, GPUImageType>;
  CastType::Pointer cast = CastType::New();
  cast->SetInput(image);

  using FilterType = itk::GPUDiscreteGaussianImageFilter<GPUImageType, GPUImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(cast->GetOutput());
  filter->SetSigma(sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  cast->Update();
  filter->Update();
  filter->GetOutput()->UpdateBuffers();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ns>(end - start);
}

ns
run_hal_cpu(ImageType * image, float sigma)
{
  using FilterType = itk::HalideDiscreteGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetVariance(sigma * sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  filter->Update();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ns>(end - start);
}

ns
run_hal_gpu(ImageType * image, float sigma)
{
  using FilterType = itk::HalideGPUDiscreteGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetVariance(sigma * sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  filter->Update();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ns>(end - start);
}

ImageType::Pointer
make_image(float extent, size_t resolution)
{
  ImageType::Pointer image = ImageType::New();

  {
    ImageType::IndexType index;
    index.Fill(0);

    ImageType::SizeType size;
    size.Fill(static_cast<ImageType::SizeValueType>(extent * static_cast<float>(resolution)));

    ImageType::RegionType region;
    region.SetIndex(index);
    region.SetSize(size);

    image->SetRegions(region);

    ImageType::SpacingType spacing;
    spacing.Fill(1.0 / static_cast<double>(resolution));

    image->SetSpacing(spacing);

    image->Allocate();
  }

  NoiseFilter::Pointer noise = NoiseFilter::New();
  noise->SetInput(image);
  noise->SetMean(0);
  noise->SetStandardDeviation(2.0);
  noise->Update();

  return noise->GetOutput();
}

size_t
get_kernel_radius(ImageType * image, float sigma)
{
  CPUBlur::Pointer temp = CPUBlur::New();
  temp->SetInput(image);
  temp->SetSigma(sigma);
  temp->SetMaximumKernelWidth(48);
  return temp->GetKernelRadius()[0];
}

double
bandwidth(ns time, size_t count)
{
  return count / time.count();
}

int
main(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " OUT" << std::endl;
    return EXIT_FAILURE;
  }

  std::string   out_path(argv[1]);
  std::ofstream csv(out_path);
  size_t        samples = 10;

  size_t image_width = 500;
  size_t byte_count = image_width * image_width * image_width * sizeof(ImageType::PixelType);

  ImageType::Pointer image = make_image(image_width, 1.0);

  // warm-up device context
  {
    run_itk_cpu(image, 1);
    run_itk_rec(image, 1);
    run_itk_gpu(image, 1);
    run_hal_cpu(image, 1);
    run_hal_cpu(image, 1);
  }

  csv << "sigma,radius,itk_cpu,itk_rec,itk_gpu,hal_cpu,hal_gpu" << std::endl;

  for (float sigma = 1; sigma < 20; sigma += 1.5)
  {
    size_t radius = get_kernel_radius(image, sigma);

    std::cout << "sigma " << sigma << " (radius " << radius << ")" << std::flush;

    for (int k = 0; k < samples; ++k)
    {
      std::cout << "." << std::flush;

      csv << sigma << "," << radius << ",";
      csv << bandwidth(run_itk_cpu(image, sigma), byte_count) << ",";
      csv << bandwidth(run_itk_rec(image, sigma), byte_count) << ",";
      csv << bandwidth(run_itk_gpu(image, sigma), byte_count) << ",";
      csv << bandwidth(run_hal_cpu(image, sigma), byte_count) << ",";
      csv << bandwidth(run_hal_gpu(image, sigma), byte_count);
      csv << std::endl;
    }

    std::cout << std::endl;
  }


  return EXIT_SUCCESS;
}
