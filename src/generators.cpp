#include "Halide.h"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Halide;

class DiscreteGaussianGenerator : public Generator<DiscreteGaussianGenerator>
{
public:
  Input<Buffer<float, 3>> input{ "input" };

  Input<float> sigma_x{ "sigma_x" };
  Input<float> sigma_y{ "sigma_y" };
  Input<float> sigma_z{ "sigma_z" };

  Output<Buffer<float, 3>> output{ "output" };

  Var x{ "x" }, y{ "y" }, z{ "z" };

  void
  generate()
  {
    using namespace ConciseCasts;

    Var i{ "i" };

    const auto root_2_pi = static_cast<float>(std::sqrt(M_PI * 2));

    Func kernel_x{ "kernel_x" };
    kernel_x(i) = Halide::exp(-i * i / (2 * sigma_x * sigma_x)) / Expr(root_2_pi) * sigma_x;

    Func kernel_y{ "kernel_y" };
    kernel_y(i) = Halide::exp(-i * i / (2 * sigma_y * sigma_y)) / Expr(root_2_pi) * sigma_y;

    Func kernel_z{ "kernel_z" };
    kernel_z(i) = Halide::exp(-i * i / (2 * sigma_z * sigma_z)) / Expr(root_2_pi) * sigma_z;

    Expr r_x = i32(2 * sigma_x + 1);
    RDom k_x{ -r_x, 2 * r_x + 1, "k_x" };

    Expr r_y = i32(2 * sigma_y + 1);
    RDom k_y{ -r_y, 2 * r_y + 1, "k_y" };

    Expr r_z = i32(2 * sigma_z + 1);
    RDom k_z{ -r_z, 2 * r_z + 1, "k_z" };

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

    output.gpu_tile(x, y, z, ix, iy, iz, ox, oy, oz, 4, 4, 4);
  }
};

HALIDE_REGISTER_GENERATOR(DiscreteGaussianGenerator, itkHalideDiscreteGaussianImpl)
