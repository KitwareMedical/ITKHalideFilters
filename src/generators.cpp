#include "Halide.h"

using namespace Halide;

class SeparableConvolutionGenerator : public Generator<SeparableConvolutionGenerator>
{
public:
  Input<Buffer<float, 3>> input{ "input" };

  Input<Buffer<float, 1>> kernel_x{"kernel_x"};
  Input<Buffer<float, 1>> kernel_y{"kernel_y"};
  Input<Buffer<float, 1>> kernel_z{"kernel_z"};

  Output<Buffer<float, 3>> output{ "output" };

  Var x{ "x" }, y{ "y" }, z{ "z" };

  void
  generate()
  {
    using namespace ConciseCasts;

    RDom k_x{ kernel_x.dim(0).min(), kernel_x.dim(0).extent(), "k_x" };
    RDom k_y{ kernel_y.dim(0).min(), kernel_y.dim(0).extent(), "k_y" };
    RDom k_z{ kernel_z.dim(0).min(), kernel_z.dim(0).extent(), "k_z" };

    Func safe = BoundaryConditions::repeat_edge(input);

    Func blur_x{ "blur_x" };
    blur_x(x, y, z) = f32(0);
    blur_x(x, y, z) += safe(x + k_x, y, z) * kernel_x(k_x);

    Func blur_y{ "blur_y" };
    blur_y(x, y, z) = f32(0);
    blur_y(x, y, z) += blur_x(x, y + k_y, z) * kernel_y(k_y);

    Func blur_z{ "blur_z" };
    blur_z(x, y, z) = f32(0);
    blur_z(x, y, z) += blur_y(x, y, z + k_z) * kernel_z(k_z);

    output(x, y, z) = blur_z(x, y, z);

    Var ix, iy, iz, ox, oy, oz;

    blur_x.compute_root();
    blur_x.update().gpu_tile(x, y, z, ix, iy, iz, ox, oy, oz, 8, 8, 8);

    blur_y.compute_root();
    blur_y.update().gpu_tile(x, y, z, ix, iy, iz, ox, oy, oz, 8, 8, 8);

    blur_z.compute_root();
    blur_z.update().gpu_tile(x, y, z, ix, iy, iz, ox, oy, oz, 8, 8, 8);

    output.compute_root();
  }
};

HALIDE_REGISTER_GENERATOR(SeparableConvolutionGenerator, itkHalideSeparableConvolutionImpl)
