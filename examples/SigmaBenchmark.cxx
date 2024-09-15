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

using ImageType = itk::Image<float, 3>;
using NoiseFilter = itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType>;
using GPUImageType = itk::GPUImage<float, 3>;
using CastToGPUImage = itk::CastImageFilter<ImageType, GPUImageType>;

using CPUBlur = itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
using HalideBlur = itk::HalideDiscreteGaussianImageFilter<ImageType, ImageType>;

using GPUBlur = itk::GPUDiscreteGaussianImageFilter<GPUImageType, GPUImageType>;
using HalideGPUBlur = itk::HalideGPUDiscreteGaussianImageFilter<ImageType, ImageType>;

using ms = std::chrono::duration<double, std::milli>;

ms
run_itk_cpu(ImageType * image, float sigma)
{
  using FilterType = itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetVariance(sigma * sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  filter->Update();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ms>(end - start);
}

ms
run_itk_gpu(ImageType * image, float sigma)
{
  using CastType = itk::CastImageFilter<ImageType, GPUImageType>;
  CastType::Pointer cast = CastType::New();
  cast->SetInput(image);

  using FilterType = itk::GPUDiscreteGaussianImageFilter<GPUImageType, GPUImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(cast->GetOutput());
  filter->SetVariance(sigma * sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  cast->Update();
  filter->Update();
  filter->GetOutput()->UpdateBuffers();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ms>(end - start);
}

ms
run_halide_cpu(ImageType * image, float sigma)
{
  using FilterType = itk::HalideDiscreteGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetVariance(sigma * sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  filter->Update();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ms>(end - start);
}

ms
run_halide_gpu(ImageType * image, float sigma)
{
  using FilterType = itk::HalideGPUDiscreteGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetVariance(sigma * sigma);
  filter->SetMaximumKernelWidth(48);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  filter->Update();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<ms>(end - start);
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

  float  extent = 300.0;
  size_t resolution = 1;

  ImageType::Pointer image = make_image(extent, resolution);

  {
    // warm-up device context
    run_itk_cpu(image, 1);
    run_itk_gpu(image, 1);
    run_halide_cpu(image, 1);
    run_halide_gpu(image, 1);
  }

  size_t samples = 5;

  csv << "sigma,itk_cpu,itk_gpu,itk_halide_cpu,itk_halide_gpu" << std::endl;

  const auto proc = [&](float sigma) {
    std::cout << "sigma " << sigma << " " << std::flush;

    for (size_t sample = 0; sample < samples; sample++)
    {
      std::cout << "." << std::flush;

      csv << sigma << ",";

      if (sigma <= 5) // ITK CPU is prohibitively slow past this point
      {
        csv << run_itk_cpu(image, sigma).count() << ",";
      }
      else
      {
        csv << "nan,";
      }

      // if (extent * res < 800) // ITK GPU memory allocation failure past this point
      // {
      csv << run_itk_gpu(image, sigma).count() << ",";
      // }
      // else
      // {
      //   csv << "nan,";
      // }

      if (sigma < 19) // Halide CPU is prohibitively slow past this point
      {
        csv << run_halide_cpu(image, sigma).count() << ",";
      }
      else
      {
        csv << "nan,";
      }

      csv << run_halide_gpu(image, sigma).count() << ",";

      csv << std::endl;
    }

    std::cout << std::endl;
  };

  for (int i = 1; i <= 8; i += 1)
  {
    proc(static_cast<float>(i));
  }
  for (int i = 10; i <= 25; i += 3)
  {
    // beyond 25, kernel is too large.
    proc(static_cast<float>(i));
  }

  return EXIT_SUCCESS;
}
