/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/image_ops.cc
//WK:Import header files..keep
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/crop_and_resize_quad_op.h"

#include <functional>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
//WK: Use this if got use gpu...keep
//#if GOOGLE_CUDA
//#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
//#include "tensorflow/core/platform/cuda.h"
//#include "tensorflow/core/platform/stream_executor.h"

//using ::perftools::gputools::cuda::ScopedActivateExecutorContext;
//#endif  // GOOGLE_CUDA
//WK: use for tensorflow(custom for tensorflow , so will see tensorflow ::  
namespace tensorflow {
//WK:what this doing???
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
using Callback = std::function<void()>;

namespace {
//WK:Use to check status on inputs, if not as required , return errors---keep
static inline Status ParseAndCheckBoxSizes(const Tensor& boxes,
                                           const Tensor& box_index,
                                           int* num_boxes) {
  if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {
    *num_boxes = 0;
    return Status::OK();
  }
  // The shape of 'boxes' is [num_boxes, 4].
  if (boxes.dims() != 2) {
    return errors::InvalidArgument("boxes must be 2-D",
                                   boxes.shape().DebugString());
  }
  *num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 4) {
    return errors::InvalidArgument("boxes must have 4 columns");
  }
  // The shape of 'box_index' is [num_boxes].
  if (box_index.dims() != 1) {
    return errors::InvalidArgument("box_index must be 1-D",
                                   box_index.shape().DebugString());
  }
  if (box_index.dim_size(0) != *num_boxes) {
    return errors::InvalidArgument("box_index has incompatible shape");
  }
  return Status::OK();
}

// Conditionally calls the compute callback if all values in box_index are in
// [0, batch_size) then calls done.
template <typename Device>
inline void RunIfBoxIndexIsValid(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done);

// Specialization of CheckValidBoxIndex for a CPUDevice.
template <>
inline void RunIfBoxIndexIsValid<CPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done) {
  const int num_boxes = box_index.dimension(0);
  for (int b = 0; b < num_boxes; ++b) {
    OP_REQUIRES_ASYNC(
        context, FastBoundsCheck(box_index(b), batch_size),
        errors::OutOfRange("box_index has values outside [0, batch_size)"),
        done);
  }
  if (compute) {
    compute();
  }
  if (done) {
    done();
  }
}

}  // namespace

template <typename Device, typename T>
class CropAndResizeQuadOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeQuadOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "biquadratic",
                errors::InvalidArgument("method must be 'biquadratic'", method));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolation_value",
                                             &extrapolation_value_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'image' is [batch_size, image_height, image_width,
    // channels].
    const Tensor& image = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'crop_size' is [2].
    const Tensor& crop_size = context->input(3);

    // Validate inputs dimensions.
    OP_REQUIRES_ASYNC(context, image.dims() == 4,
                      errors::InvalidArgument("input image must be 4-D",
                                              image.shape().DebugString()),
                      done);
    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    const int depth = image.dim_size(3);
    OP_REQUIRES_ASYNC(
        context, image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);

    OP_REQUIRES_ASYNC(context, crop_size.dims() == 1,
                      errors::InvalidArgument("crop_size must be 1-D",
                                              crop_size.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(
        context, crop_size.dim_size(0) == 2,
        errors::InvalidArgument("crop_size must have two elements",
                                crop_size.shape().DebugString()),
        done);

    // Copy and validate crop sizes.
    auto crop_size_vec = crop_size.vec<int32>();
    const int crop_height = internal::SubtleMustCopy(crop_size_vec(0));
    const int crop_width = internal::SubtleMustCopy(crop_size_vec(1));
    OP_REQUIRES_ASYNC(
        context, crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("crop dimensions must be positive"), done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({num_boxes, crop_height, crop_width, depth}),
            &output),
        done);

    auto compute_callback = [this, context, output]() {
      const Tensor& image = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const bool status = functor::CropAndResizeQuad<Device, T>()(
          context, image.tensor<T, 4>(), boxes.tensor<float, 2>(),
          box_index.tensor<int32, 1>(), extrapolation_value_,
          output->tensor<float, 4>());
      if (!status) {
        context->SetStatus(
            errors::Internal("Failed launch CropAndResizeQuadKernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }

 private:
  float extrapolation_value_;
};

// Partial specialization of CropAndResizeQuad functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResizeQuad<CPUDevice, T> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  float extrapolation_value,
                  typename TTypes<float, 4>::Tensor crops) {
    const int batch_size = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = crops.dimension(0);
    const int crop_height = crops.dimension(1);
    const int crop_width = crops.dimension(2);
    const int depth = crops.dimension(3);

    // Sharding across boxes.
    auto CropAndResizeQuadPerBox = [&](int start_box, int limit_box) {
      for (int b = start_box; b < limit_box; ++b) {
        const float y1 = boxes(b, 0);
        const float x1 = boxes(b, 1);
        const float y2 = boxes(b, 2);
        const float x2 = boxes(b, 3);

        const int32 b_in = box_index(b);
        if (!FastBoundsCheck(b_in, batch_size)) {
          continue;
        }

        const float height_scale =
            (crop_height > 1)
                ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1)
                             : 0;

        for (int y = 0; y < crop_height; ++y) {
          const float in_y = (crop_height > 1)
                                 ? y1 * (image_height - 1) + y * height_scale
                                 : 0.5 * (y1 + y2) * (image_height - 1);
          if (in_y < 0 || in_y > image_height - 1) {
            for (int x = 0; x < crop_width; ++x) {
              for (int d = 0; d < depth; ++d) {
                crops(b, y, x, d) = extrapolation_value;
              }
            }
            continue;
          }
          const int mid_mid_y= floorf(in_y);
          const int top_mid_y= floorf(in_y)-1;
          const int bot_mid_y= ceilf(in_y);
          const float y_lerp = in_y - mid_mid_y;

          for (int x = 0; x < crop_width; ++x) {
            const float in_x = (crop_width > 1)
                                   ? x1 * (image_width - 1) + x * width_scale
                                   : 0.5 * (x1 + x2) * (image_width - 1);
            if (in_x < 0 || in_x > image_width - 1) {
              for (int d = 0; d < depth; ++d) {
                crops(b, y, x, d) = extrapolation_value;
              }
              continue;
            }
            const int mid_mid_x = floorf(in_x);
            const int left_mid_x = floorf(in_x)-1;
            const int right_mid_x = ceilf(in_x);
            const float x_lerp = in_x - mid_mid_x;

            for (int d = 0; d < depth; ++d) {
              const float top_left(static_cast<float>(
                  image(b_in, top_mid_y, left_mid_x, d)));
              const float top_mid(static_cast<float>(
                  image(b_in, top_mid_y, mid_mid_x, d)));
              const float top_right(static_cast<float>(
                  image(b_in, top_mid_y, right_mid_x, d)));
                  
              const float mid_left(static_cast<float>(
                  image(b_in, mid_mid_y, left_mid_x, d)));
              const float top_mid(static_cast<float>(
                  image(b_in, mid_mid_y, mid_mid_x, d)));
              const float mid_right(static_cast<float>(
                  image(b_in, mid_mid_y, right_mid_x, d)));
                  
             const float bot_left(static_cast<float>(
                  image(b_in,bot_mid_y, left_mid_x, d)));
             const float bot_mid(static_cast<float>(
                  image(b_in, bot_mid_y, mid_mid_x, d)));
             const float bot_right(static_cast<float>(
                  image(b_in, bot_mid_y, right_mid_x, d)));
      
              const float CA = (top_mid)+(top_right - top_left)*(x_lerp)+(top_left - (2*top_mid)+top_right)*pow(x_lerp,2)
              const float CB = (mid_mid)+(mid_right – mid_left)*(x_lerp)+(mid_left - (2*mid_mid)+mid_right)*pow(x_lerp,2)
              const float CC = (bot_mid)+(bot_right – mid_left)*(x_lerp)+(bot_left - (2*bot_mid)+bot_right)*pow(x_lerp,2)
              const float crops(b,y,x,d) = (CB)+(CC– CA)*(y_lerp)+(CA - (2*CB)+CC)*pow(y_lerp,2)

            }
          }
        }
      }
    };

    // A rough estimation of the cost for each cropped box.
    const double cost_per_pixel =
        depth * (Eigen::TensorOpCost::AddCost<float>() * 6 +
                 Eigen::TensorOpCost::MulCost<float>() * 3 +
                 Eigen::TensorOpCost::CastCost<T, float>() * 4) +
        (Eigen::TensorOpCost::AddCost<float>() * 2 +
         Eigen::TensorOpCost::AddCost<float>() * 3);
    const double cost_per_box = crop_height * crop_width * cost_per_pixel;

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_boxes,
          cost_per_box, CropAndResizeQuadPerBox);

    return true;
  }
};

}  // namespace functor

template <typename Device, typename T>
class CropAndResizeQuadGradImageOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeQuadGradImageOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "biquadratic",
                errors::InvalidArgument("method must be 'biquadratic'", method));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'image_size' is [4].
    const Tensor& image_size = context->input(3);

    // Validate input shapes.
    OP_REQUIRES_ASYNC(context, grads.dims() == 4,
                      errors::InvalidArgument("grads image must be 4-D",
                                              grads.shape().DebugString()),
                      done);
    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    OP_REQUIRES_ASYNC(
        context, crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("grads dimensions must be positive"), done);
    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);
    OP_REQUIRES_ASYNC(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"),
        done);

    OP_REQUIRES_ASYNC(context, image_size.dims() == 1,
                      errors::InvalidArgument("image_size must be 1-D",
                                              image_size.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(context, image_size.dim_size(0) == 4,
                      errors::InvalidArgument("image_size must have 4 elements",
                                              image_size.shape().DebugString()),
                      done);
    auto image_size_vec = image_size.vec<int32>();
    const int batch_size = internal::SubtleMustCopy(image_size_vec(0));
    const int image_height = internal::SubtleMustCopy(image_size_vec(1));
    const int image_width = internal::SubtleMustCopy(image_size_vec(2));
    const int depth = internal::SubtleMustCopy(image_size_vec(3));
    OP_REQUIRES_ASYNC(
        context, image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    OP_REQUIRES_ASYNC(
        context, grads.dim_size(3) == depth,
        errors::InvalidArgument("image_size and grads are incompatible"), done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, image_height, image_width, depth}),
            &output),
        done);

    auto compute_callback = [context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const bool status = functor::CropAndResizeQuadBackpropImage<Device, T>()(
          context->eigen_device<Device>(), grads.tensor<float, 4>(),
          boxes.tensor<float, 2>(), box_index.tensor<int32, 1>(),
          output->tensor<T, 4>());
      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed launch CropAndResizeQuadBackpropImage kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }
};

// Partial specialization of CropAndResizeQuadBackpropImage functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResizeQuadBackpropImage<CPUDevice, T> {
  bool operator()(const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  typename TTypes<T, 4>::Tensor grads_image) {
    const int batch_size = grads_image.dimension(0);
    const int image_height = grads_image.dimension(1);
    const int image_width = grads_image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);

    grads_image.setZero();

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxes(b, 0);
      const float x1 = boxes(b, 1);
      const float y2 = boxes(b, 2);
      const float x2 = boxes(b, 3);

      const int32 b_in = box_index(b);
      if (!FastBoundsCheck(b_in, batch_size)) {
        continue;
      }

      const float height_scale =
          (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                            : 0;
      const float width_scale =
          (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1)
                           : 0;

      for (int y = 0; y < crop_height; ++y) {
        const float in_y = (crop_height > 1)
                               ? y1 * (image_height - 1) + y * height_scale
                               : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1) {
          continue;
        }
        const int mid_mid_y= floorf(in_y);
        const int top_mid_y= floorf(in_y)-1;
        const int bot_mid_y= ceilf(in_y);
        const float y_lerp = in_y - mid_mid_y
        for (int x = 0; x < crop_width; ++x) {
           const float in_x = (crop_width > 1)
                                 ? x1 * (image_width - 1) + x * width_scale
                                 : 0.5 * (x1 + x2) * (image_width - 1);
          if (in_x < 0 || in_x > image_width - 1) {
            continue;
          }
          const int mid_mid_x = floorf(in_x);
          const int left_mid_x = floorf(in_x)-1;
          const int right_mid_x = ceilf(in_x);
          const float x_lerp = in_x - mid_mid_x;
;

          for (int d = 0; d < depth; ++d) {
            const float dgrad = grads(b, y, x, d);
            const float xsqua = pow(x_lerp ,2);
            const float ysqua = pow(y_lerp,2);
            grads_image(b_in, top_mid_y, left_mid_x, d) +=
                static_cast<T>(0.25*dgrad*(x_lerp * y_lerp –(xsqua*y_lerp)-(x_lerp*ysqua)+(xsqua*ysqua)));
            grads_image(b_in, top_mid_y, mid_mid_x, d) +=
                static_cast<T>(0.5*dgrad*((1+ 0.5*x_lerp - xsqua) + ysqua*(1 + 0.5*x_lerp - xsqua)));
            grads_image(b_in, top_mid_y, right_mid_x,d)+=
                static_cast<T>(0.25*dgrad*(xsqua*ysqua –xsqua*y_lerp));

            grads_image(b_in, mid_mid_y, left_mid_x, d) +=
                static_cast<T>(dgrad*(ysqua*(1+0.5*x_lerp -xsqua)-(0.5*(x_lerp + xsqua – x_lerp*ysqua))));
            grads_image(b_in, mid_mid_y, mid_mid_x, d) +=
                static_cast<T>(dgrad*((1+ 0.5*x_lerp – xsqua +0.5*x_lerp*ysqua) + ysqua*(1 + 0.5*x_lerp - xsqua)));
            grads_image(b_in, mid_mid_y, right_mid_x,d)+=
                static_cast<T>(0.5*dgrad*(xsqua*ysqua  + xsqua));

           grads_image(b_in, bot_mid_y, left_mid_x, d) +=
                static_cast<T>(0.25*dgrad*(xsqua*ysqua + xsqua*ylerp – x_lerp*y_lerp));
            grads_image(b_in, bot_mid_y, mid_mid_x, d) +=
                static_cast<T>(0.5*dgrad*(y_lerp*(1 + 0.5*x_lerp - xsqua) + ysqua(1+0.5*x_lerp -xsqua)));
            grads_image(b_in, bot_mid_y, right_mid_x,d)+=
                static_cast<T>(0.25*dgrad*(xsqua*ysqua  + xsqua*y_lerp));

            
          }
        }
      }
    }
    return true;
  }
};

}  // namespace functor

template <typename Device, typename T>
class CropAndResizeQuadGradBoxesOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeQuadGradBoxesOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "biquadratic",
                errors::InvalidArgument("method must be 'biquadratic'", method));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(2);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(3);
    // The shape of 'image' is [batch_size, image_height, image_width, depth].
    const Tensor& image = context->input(1);

    // Validate input shapes.
    OP_REQUIRES_ASYNC(context, grads.dims() == 4,
                      errors::InvalidArgument("grads image must be 4-D",
                                              grads.shape().DebugString()),
                      done);
    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    const int depth = grads.dim_size(3);
    OP_REQUIRES_ASYNC(
        context, crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("grads dimensions must be positive"), done);

    OP_REQUIRES_ASYNC(context, image.dims() == 4,
                      errors::InvalidArgument("input image must be 4-D",
                                              image.shape().DebugString()),
                      done);
    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    OP_REQUIRES_ASYNC(
        context, image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    OP_REQUIRES_ASYNC(context, image.dim_size(3) == depth,
                      errors::InvalidArgument("image, grads depth differ"),
                      done);

    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);

    OP_REQUIRES_ASYNC(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"),
        done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(0, TensorShape({num_boxes, 4}), &output),
        done);

    auto compute_callback = [context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& image = context->input(1);
      const Tensor& boxes = context->input(2);
      const Tensor& box_index = context->input(3);
      const bool status = functor::CropAndResizeQuadBackpropBoxes<Device, T>()(
          context->eigen_device<Device>(), grads.tensor<float, 4>(),
          image.tensor<T, 4>(), boxes.tensor<float, 2>(),
          box_index.tensor<int32, 1>(), output->tensor<float, 2>());
      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed launch CropAndResizeQuadBackpropBoxes kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }
};

// Partial specialization of CropAndResizeQuadBackpropBoxes functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResizeQuadBackpropBoxes<CPUDevice, T> {
  bool operator()(const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  typename TTypes<float, 2>::Tensor grads_boxes) {
    const int batch_size = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);

    grads_boxes.setZero();

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxes(b, 0);
      const float x1 = boxes(b, 1);
      const float y2 = boxes(b, 2);
      const float x2 = boxes(b, 3);

      const int32 b_in = box_index(b);
      if (!FastBoundsCheck(b_in, batch_size)) {
        continue;
      }

      const float height_ratio =
          (crop_height > 1)
              ? static_cast<float>(image_height - 1) / (crop_height - 1)
              : 0;
      const float width_ratio =
          (crop_width > 1)
              ? static_cast<float>(image_width - 1) / (crop_width - 1)
              : 0;

      const float height_scale =
          (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
      const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;

      for (int y = 0; y < crop_height; ++y) {
        const float in_y = (crop_height > 1)
                               ? y1 * (image_height - 1) + y * height_scale
                               : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1) {
          continue;
        }
        const int mid_mid_y= floorf(in_y);
        const int top_mid_y= floorf(in_y)-1;
        const int bot_mid_y= ceilf(in_y);
        const float y_lerp = in_y - mid_mid_y;
        for (int x = 0; x < crop_width; ++x) {
          const float in_x = (crop_width > 1)
                                 ? x1 * (image_width - 1) + x * width_scale
                                 : 0.5 * (x1 + x2) * (image_width - 1);
          if (in_x < 0 || in_x > image_width - 1) {
            continue;
          }
           const int mid_mid_x = floorf(in_x);
           const int left_mid_x = floorf(in_x)-1;
           const int right_mid_x = ceilf(in_x);
           const float x_lerp = in_x - mid_mid_x;

          for (int d = 0; d < depth; ++d) {
              const float top_left(
                       static_cast<float>(image(b_in, top_mid_y, left_mid_x, d)));
              const float top_mid(
                       static_cast<float>(image(b_in, top_mid_y, mid_mid_x, d)));
              const float top_right(
                       static_cast<float>(image(b_in, top_mid_y, right_mid_x, d)));

              const float mid_left(
                       static_cast<float>(image(b_in, mid_mid_y, left_mid_x, d)));
              const float mid_mid(
                       static_cast<float>(image(b_in, mid_mid_y, mid_mid_x, d)));
              const float mid_right(
                       static_cast<float>(image(b_in, mid_mid_y, right_mid_x, d)));
                  
             const float bot_left(
                       static_cast<float>(image(b_in,bot_mid_y, left_mid_x, d)));
             const float bot_mid(
                       static_cast<float>(image(b_in, bot_mid_y, mid_mid_x, d)));
             const float bot_right(
                      static_cast<float>(image(b_in, bot_mid_y, right_mid_x, d)));

//compute the image gradient
            float dA_x = 0.5*(top_mid - top_left) + x_lerp*(top_left - 2*top_mid + top_right);
            float dB_x = 0.5(mid_mid - mid_left) + x_lerp*(mid_left - 2*mid_mid + mid_right);
            float dC_x = 0.5*(bot_mid - bot_left) + x_lerp*(bot_left - 2*bot_mid + bot_right);
            CA = (top_mid)+(top_right - top_left)*(x_lerp)+(top_left - (*top_mid)+top_right)*pow(x_lerp,2);
            CB = (mid_mid)+(mid_right – mid_left)*(x_lerp)+(mid_left - (2*mid_mid)+mid_right)*pow(x_lerp,2);
            CC = (bot_mid)+(bot_right – mid_left)*(x_lerp)+(bot_left - (2*bot_mid)+bot_right)*pow(x_lerp,2);
           float  image_grad_x =  dB_x +0.5* (dC_x - dA_x)*y_lerp + 0.5*(dA_x + 2*dB_x + dC_x)*(pow(y_lerp,2));
           float image_grad_y = 0.5*(CC - CA) + (CA + 2*CB +CC)*(y_lerp);



            // Modulate the image gradient with the incoming gradient.
            const float top_grad = grads(b, y, x, d);
            image_grad_y *= top_grad;
            image_grad_x *= top_grad;
            // dy1, dy2
            if (crop_height > 1) {
              grads_boxes(b, 0) +=
                  image_grad_y * (image_height - 1 - y * height_ratio);
              grads_boxes(b, 2) += image_grad_y * (y * height_ratio);
            } else {
              grads_boxes(b, 0) += image_grad_y * 0.5 * (image_height - 1);
              grads_boxes(b, 2) += image_grad_y * 0.5 * (image_height - 1);
            }
            // dx1, dx2
            if (crop_width > 1) {
              grads_boxes(b, 1) +=
                  image_grad_x * (image_width - 1 - x * width_ratio);
              grads_boxes(b, 3) += image_grad_x * (x * width_ratio);
            } else {
              grads_boxes(b, 1) += image_grad_x * 0.5 * (image_width - 1);
              grads_boxes(b, 3) += image_grad_x * 0.5 * (image_width - 1);
            }
          }
        }
      }
    }
    return true;
  }
};

}  // namespace functor

#define REGISTER_KERNEL(T)                                \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeQuad")           \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T")     \
                              .HostMemory("crop_size"),   \
                          CropAndResizeQuadOp<CPUDevice, T>); \
                                                          \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeQuadGradBoxes")  \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T"),    \
                          CropAndResizeQuadGradBoxesOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                               \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeQuadGradImage") \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("image_size"), \
                          CropAndResizeQuadGradImageOp<CPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);


