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

#include "itkHalideCastImageFilter.h"

#include "itkCommand.h"
#include "itkTestingMacros.h"
#include "itkImageRegionConstIterator.h"

namespace
{
class ShowProgress : public itk::Command
{
public:
  itkNewMacro(ShowProgress);

  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * caller, const itk::EventObject & event) override
  {
    if (!itk::ProgressEvent().CheckEvent(&event))
    {
      return;
    }
    const auto * processObject = dynamic_cast<const itk::ProcessObject *>(caller);
    if (!processObject)
    {
      return;
    }
    std::cout << " " << processObject->GetProgress();
  }
};
} // namespace

int
itkHalideCastImageFilterTest(int argc, char * argv[])
{
  constexpr unsigned int Dimension = 2;
  using InputPixelType = float;
  using InputImageType = itk::Image<InputPixelType, Dimension>;
  using OutputPixelType = uint32_t;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  using FilterType = itk::HalideCastImageFilter<InputImageType, OutputImageType>;
  FilterType::Pointer filter = FilterType::New();

  ITK_EXERCISE_BASIC_OBJECT_METHODS(filter, HalideCastImageFilter, ImageToImageFilter);

  // Create input image to avoid test dependencies.
  InputImageType::SizeType size;
  size.Fill(128);
  InputImageType::Pointer image = InputImageType::New();
  image->SetRegions(size);
  image->Allocate();
  image->FillBuffer(1.1f);

  ShowProgress::Pointer showProgress = ShowProgress::New();
  filter->AddObserver(itk::ProgressEvent(), showProgress);
  filter->SetInput(image);

  ITK_TRY_EXPECT_NO_EXCEPTION(filter->Update());

  OutputImageType * outputImage = filter->GetOutput();

  using OutputRegionType = typename OutputImageType::RegionType;
  OutputRegionType outputRegion = OutputRegionType(size);

  itk::ImageRegionConstIterator<OutputImageType> out(outputImage, outputRegion);

  for (out.GoToBegin(); !out.IsAtEnd(); ++out)
  {
    assert(out.Get() == 1);
  }

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}
