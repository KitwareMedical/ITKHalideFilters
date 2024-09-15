#include "Halide.h"

using namespace Halide;

class SeparableConvolutionGenerator : public Generator<SeparableConvolutionGenerator>
{
public:
  GeneratorParam<bool> use_gpu{ "use_gpu", true };

  Input<Buffer<float, 3>> input{ "input" };
  Input<Buffer<float, 1>> kernel_x{ "kernel_x" };
  Input<Buffer<float, 1>> kernel_y{ "kernel_y" };
  Input<Buffer<float, 1>> kernel_z{ "kernel_z" };

  Output<Buffer<float, 3>> output{ "output" };

  Var  x{ "x" }, y{ "y" }, z{ "z" };
  Func blur_x{ "blur_x" }, blur_y{ "blur_y" }, blur_z{ "blur_z" };
  Func sample{ "sample" };

  void
  generate()
  {
    using namespace ConciseCasts;

    RDom k_x{ kernel_x.dim(0).min(), kernel_x.dim(0).extent(), "k_x" };
    RDom k_y{ kernel_y.dim(0).min(), kernel_y.dim(0).extent(), "k_y" };
    RDom k_z{ kernel_z.dim(0).min(), kernel_z.dim(0).extent(), "k_z" };

    // zero-flux boundary condition
    sample = BoundaryConditions::repeat_edge(input);

    blur_x(x, y, z) = f32(0);
    blur_x(x, y, z) += sample(x + k_x, y, z) * kernel_x(k_x);

    blur_y(x, y, z) = f32(0);
    blur_y(x, y, z) += blur_x(x, y + k_y, z) * kernel_y(k_y);

    blur_z(x, y, z) = f32(0);
    blur_z(x, y, z) += blur_y(x, y, z + k_z) * kernel_z(k_z);

    output(x, y, z) = blur_z(x, y, z);

    if (using_autoscheduler())
    {
      input.set_estimates({ { 0, 300 }, { 0, 300 }, { 0, 300 } });
      output.set_estimates({ { 0, 300 }, { 0, 300 }, { 0, 300 } });
      kernel_x.set_estimates({ { -10, 10 } });
      kernel_y.set_estimates({ { -10, 10 } });
      kernel_z.set_estimates({ { -10, 10 } });
    }
    else if (use_gpu)
    {
      schedule_gpu();
    }
    else
    {
      schedule_cpu();
    }
  }

  /**
   * Schedule using precomputed autoschedule. Obtained with Adams2019 using:
   * - Input/Output size estimate 300x300x300
   * - Kernel size estimate 21
   * - Intel i9-14900k
   */
  void
  schedule_cpu()
  {
    using Halide::_0;
    using Halide::_1;
    using Halide::_2;

    Var  _0i("_0i");
    Var  xi("xi");
    Var  xii("xii");
    Var  yi("yi");
    Var  yii("yii");
    Var  zi("zi");
    Var  zii("zii");
    RVar k_x_x(blur_x.update(0).get_schedule().dims()[0].var);
    RVar k_y_x(blur_y.update(0).get_schedule().dims()[0].var);
    RVar k_z_x(blur_z.update(0).get_schedule().dims()[0].var);
    output.split(y, y, yi, 38, TailStrategy::ShiftInwards)
      .split(z, z, zi, 38, TailStrategy::ShiftInwards)
      .split(x, x, xi, 24, TailStrategy::ShiftInwards)
      .split(yi, yi, yii, 2, TailStrategy::ShiftInwards)
      .split(zi, zi, zii, 2, TailStrategy::ShiftInwards)
      .split(xi, xi, xii, 8, TailStrategy::ShiftInwards)
      .unroll(xi)
      .unroll(yii)
      .unroll(zii)
      .vectorize(xii)
      .compute_root()
      .reorder({ xii, xi, yii, zii, zi, yi, x, y, z })
      .fuse(y, z, y)
      .parallel(y);
    blur_z.store_in(MemoryType::Stack)
      .split(x, x, xi, 8, TailStrategy::RoundUp)
      .unroll(x)
      .unroll(y)
      .unroll(z)
      .vectorize(xi)
      .compute_at(output, zi)
      .reorder({ xi, x, y, z });
    blur_z.update(0)
      .split(x, x, xi, 8, TailStrategy::GuardWithIf)
      .unroll(x)
      .unroll(y)
      .unroll(z)
      .vectorize(xi)
      .reorder({ xi, x, y, z, k_z_x });
    blur_y.store_in(MemoryType::Stack)
      .split(x, x, xi, 8, TailStrategy::RoundUp)
      .vectorize(xi)
      .compute_at(output, yi)
      .reorder({ xi, x, y, z });
    blur_y.update(0).split(x, x, xi, 8, TailStrategy::GuardWithIf).vectorize(xi).reorder({ xi, k_y_x, x, y, z });
    blur_x.split(x, x, xi, 8, TailStrategy::RoundUp).vectorize(xi).compute_at(output, x).reorder({ xi, x, y, z });
    blur_x.update(0)
      .split(x, x, xi, 8, TailStrategy::GuardWithIf)
      .unroll(x)
      .vectorize(xi)
      .reorder({ xi, x, k_x_x, y, z });
    sample.store_in(MemoryType::Stack)
      .split(_0, _0, _0i, 8, TailStrategy::ShiftInwards)
      .vectorize(_0i)
      .compute_at(blur_x, y)
      .reorder({ _0i, _0, _1, _2 });
  }

  /**
   * Schedule using precomputed autoschedule. Obtained with Anderson2021 using:
   * - Input/Output size estimate 300x300x300
   * - Kernel size estimate 21
   * - CUDA
   * - Nvidia RTX 4090
   */
  void
  schedule_gpu()
  {
    Var  xi("xi");
    Var  yi("yi");
    Var  yii("yii");
    Var  zi("zi");
    Var  zii("zii");
    RVar k_x_x(blur_x.update(0).get_schedule().dims()[0].var);
    RVar k_y_x(blur_y.update(0).get_schedule().dims()[0].var);
    RVar k_z_x(blur_z.update(0).get_schedule().dims()[0].var);
    Var  zi_serial_outer("zi_serial_outer");
    Var  yi_serial_outer("yi_serial_outer");
    Var  xi_serial_outer("xi_serial_outer");
    output.split(x, x, xi, 16, TailStrategy::ShiftInwards)
      .split(y, y, yi, 4, TailStrategy::ShiftInwards)
      .split(z, z, zi, 4, TailStrategy::ShiftInwards)
      .split(yi, yi, yii, 2, TailStrategy::ShiftInwards)
      .unroll(yii)
      .compute_root()
      .reorder(yii, xi, yi, zi, x, y, z)
      .gpu_blocks(x)
      .gpu_blocks(y)
      .gpu_blocks(z)
      .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
      .gpu_threads(xi)
      .split(yi, yi_serial_outer, yi, 2, TailStrategy::GuardWithIf)
      .gpu_threads(yi)
      .split(zi, zi_serial_outer, zi, 4, TailStrategy::GuardWithIf)
      .gpu_threads(zi);
    blur_z.split(x, x, xi, 16, TailStrategy::RoundUp)
      .split(y, y, yi, 2, TailStrategy::RoundUp)
      .split(z, z, zi, 16, TailStrategy::RoundUp)
      .split(yi, yi, yii, 2, TailStrategy::RoundUp)
      .split(zi, zi, zii, 4, TailStrategy::RoundUp)
      .unroll(yii)
      .unroll(zii)
      .compute_root()
      .reorder(yii, zii, xi, yi, zi, x, y, z)
      .gpu_blocks(x)
      .gpu_blocks(y)
      .gpu_blocks(z)
      .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
      .gpu_threads(xi)
      .split(zi, zi_serial_outer, zi, 4, TailStrategy::GuardWithIf)
      .gpu_threads(zi);
    blur_z.update(0)
      .split(x, x, xi, 16, TailStrategy::GuardWithIf)
      .split(y, y, yi, 2, TailStrategy::GuardWithIf)
      .split(z, z, zi, 16, TailStrategy::GuardWithIf)
      .split(yi, yi, yii, 2, TailStrategy::GuardWithIf)
      .split(zi, zi, zii, 4, TailStrategy::GuardWithIf)
      .unroll(yii)
      .unroll(zii)
      .reorder(yii, zii, k_z_x, xi, yi, zi, x, y, z)
      .gpu_blocks(x)
      .gpu_blocks(y)
      .gpu_blocks(z)
      .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
      .gpu_threads(xi)
      .split(zi, zi_serial_outer, zi, 4, TailStrategy::GuardWithIf)
      .gpu_threads(zi);
    blur_y.split(x, x, xi, 16, TailStrategy::RoundUp)
      .split(y, y, yi, 16, TailStrategy::RoundUp)
      .split(z, z, zi, 2, TailStrategy::RoundUp)
      .split(yi, yi, yii, 4, TailStrategy::RoundUp)
      .split(zi, zi, zii, 2, TailStrategy::RoundUp)
      .unroll(yii)
      .unroll(zii)
      .compute_root()
      .reorder(yii, zii, xi, yi, zi, x, y, z)
      .gpu_blocks(x)
      .gpu_blocks(y)
      .gpu_blocks(z)
      .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
      .gpu_threads(xi)
      .split(yi, yi_serial_outer, yi, 4, TailStrategy::GuardWithIf)
      .gpu_threads(yi);
    blur_y.update(0)
      .split(x, x, xi, 16, TailStrategy::GuardWithIf)
      .split(y, y, yi, 16, TailStrategy::GuardWithIf)
      .split(z, z, zi, 2, TailStrategy::GuardWithIf)
      .split(yi, yi, yii, 4, TailStrategy::GuardWithIf)
      .split(zi, zi, zii, 2, TailStrategy::GuardWithIf)
      .unroll(yii)
      .unroll(zii)
      .reorder(yii, zii, k_y_x, xi, yi, zi, x, y, z)
      .gpu_blocks(x)
      .gpu_blocks(y)
      .gpu_blocks(z)
      .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
      .gpu_threads(xi)
      .split(yi, yi_serial_outer, yi, 4, TailStrategy::GuardWithIf)
      .gpu_threads(yi);
    blur_x.split(x, x, xi, 16, TailStrategy::RoundUp)
      .split(y, y, yi, 8, TailStrategy::RoundUp)
      .split(z, z, zi, 2, TailStrategy::RoundUp)
      .split(yi, yi, yii, 4, TailStrategy::RoundUp)
      .split(zi, zi, zii, 2, TailStrategy::RoundUp)
      .unroll(yii)
      .unroll(zii)
      .compute_root()
      .reorder(yii, zii, xi, yi, zi, x, y, z)
      .gpu_blocks(x)
      .gpu_blocks(y)
      .gpu_blocks(z)
      .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
      .gpu_threads(xi)
      .split(yi, yi_serial_outer, yi, 2, TailStrategy::GuardWithIf)
      .gpu_threads(yi);
    blur_x.update(0)
      .split(x, x, xi, 16, TailStrategy::GuardWithIf)
      .split(y, y, yi, 8, TailStrategy::GuardWithIf)
      .split(z, z, zi, 2, TailStrategy::GuardWithIf)
      .split(yi, yi, yii, 4, TailStrategy::GuardWithIf)
      .split(zi, zi, zii, 2, TailStrategy::GuardWithIf)
      .unroll(yii)
      .unroll(zii)
      .reorder(yii, zii, k_x_x, xi, yi, zi, x, y, z)
      .gpu_blocks(x)
      .gpu_blocks(y)
      .gpu_blocks(z)
      .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
      .gpu_threads(xi)
      .split(yi, yi_serial_outer, yi, 2, TailStrategy::GuardWithIf)
      .gpu_threads(yi);
    blur_z.in(output)
      .store_in(MemoryType::Register)
      .compute_at(output, xi)
      .bound_extent(x, 1)
      .unroll(x)
      .bound_extent(y, 2)
      .unroll(y)
      .bound_extent(z, 1)
      .unroll(z);
  }
};

HALIDE_REGISTER_GENERATOR(SeparableConvolutionGenerator, itkHalideSeparableConvolutionImpl)
