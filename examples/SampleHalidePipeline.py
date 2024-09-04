#!/usr/bin/env python

import argparse
from time import perf_counter

import itk


parser = argparse.ArgumentParser(description="Example short description.")
parser.add_argument("input_image")
parser.add_argument("sigma", type=float)
parser.add_argument("output_image")

ImageType = itk.Image[itk.F, 3]
FilterType = itk.HalideDiscreteGaussianImageFilter[ImageType, ImageType]

if __name__ == '__main__':
    args = parser.parse_args()

    im = itk.imread(args.input_image, itk.F)

    proc = FilterType.New()
    proc.SetInput(im)
    proc.SetSigma(args.sigma)

    start = perf_counter()
    proc.Update()
    end = perf_counter()

    delta = end - start

    print(f'took {delta * 1_000:.3f}ms')

    itk.imwrite(proc.GetOutput(), args.output_image)
