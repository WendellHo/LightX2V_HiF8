#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cmath>
#include <string>
#include <type_traits>

#include "utils.h"

namespace {

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
  return static_cast<scalar_t>(v);
}

__device__ __forceinline__ float hif8_decode_u8(uint8_t code) {
  const int sign = (code >> 7) & 0x1;
  const int rem = code & 0x7F;

  int d = -1;
  int dot_len = 0;
  if ((rem & 0b1100000) == 0b1100000) {         // 11
    d = 4;
    dot_len = 2;
  } else if ((rem & 0b1100000) == 0b1000000) {  // 10
    d = 3;
    dot_len = 2;
  } else if ((rem & 0b1100000) == 0b0100000) {  // 01
    d = 2;
    dot_len = 2;
  } else if ((rem & 0b1110000) == 0b0010000) {  // 001
    d = 1;
    dot_len = 3;
  } else if ((rem & 0b1111000) == 0b0001000) {  // 0001
    d = 0;
    dot_len = 4;
  } else if ((rem & 0b1111000) == 0b0000000) {  // 0000 -> DML
    const int mant = rem & 0x7;
    if (mant == 0) {
      return sign ? NAN : 0.0f;
    }
    float v = exp2f(static_cast<float>(mant - 23));
    return sign ? -v : v;
  } else {
    return NAN;
  }

  const int mant_width = (d <= 2) ? 3 : ((d == 3) ? 2 : 1);
  const int tail_bits = 7 - dot_len;
  const int payload = rem & ((1 << tail_bits) - 1);
  const int exp_bits = (d > 0) ? (payload >> mant_width) : 0;
  const int mant = payload & ((1 << mant_width) - 1);

  int e_dec = 0;
  if (d > 0) {
    const int se = (exp_bits >> (d - 1)) & 0x1;
    const int mag_tail = (d > 1) ? (exp_bits & ((1 << (d - 1)) - 1)) : 0;
    const int mag = (1 << (d - 1)) | mag_tail;
    e_dec = se ? -mag : mag;
  }

  if (e_dec == 15 && mant == ((1 << mant_width) - 1)) {
    return sign ? -CUDART_INF_F : CUDART_INF_F;
  }

  const float signif = 1.0f + static_cast<float>(mant) / static_cast<float>(1 << mant_width);
  const float v = signif * exp2f(static_cast<float>(e_dec));
  return sign ? -v : v;
}

__device__ __forceinline__ float hif8_qdq_scalar(float x, float eps) {
  (void)eps;
  if (isnan(x) || isinf(x)) {
    return x;
  }

  const float x_unsigned = fabsf(x);
  const float sign = x >= 0.0f ? 1.0f : -1.0f;
  if (x_unsigned < exp2f(-23.0f)) {
    return 0.0f;
  }
  if (x_unsigned >= exp2f(15.0f) * 1.25f) {
    return sign * CUDART_INF_F;
  }

  float e = floorf(log2f(x_unsigned));
  if (e <= -23.0f) {
    e = -22.0f;
  }

  float abse = fabsf(e);
  float mant_bits = 0.0f;
  if (abse <= 15.0f) {
    mant_bits = 1.0f;
  }
  if (abse <= 7.0f) {
    mant_bits = 2.0f;
  }
  if (abse <= 3.0f) {
    mant_bits = 3.0f;
  }
  float q = floorf(x_unsigned * exp2f(-e + mant_bits) + 0.5f);
  return q * exp2f(e - mant_bits) * sign;
}

template <typename scalar_t>
__global__ void hif8_fused_mm_kernel(
    scalar_t* out,
    const scalar_t* input,
    const uint8_t* weight_u8,
    const scalar_t* bias,
    int64_t m,
    int64_t n,
    int64_t k,
    bool enable_input_qdq,
    bool enable_output_requant) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  const float eps = std::is_same<scalar_t, c10::Half>::value ? exp2f(-14.0f) : exp2f(-45.0f);

  for (int kk = 0; kk < k; ++kk) {
    float a = to_float(input[row * k + kk]);
    if (enable_input_qdq) {
      a = hif8_qdq_scalar(a, eps);
    }

    // weight_u8 shape: [k, n] using native HiF8 byte codes
    float w = hif8_decode_u8(weight_u8[kk * n + col]);

    acc += a * w;
  }

  if (bias != nullptr) {
    acc += to_float(bias[col]);
  }
  if (enable_output_requant) {
    acc = hif8_qdq_scalar(acc, eps);
  }
  out[row * n + col] = from_float<scalar_t>(acc);
}

}  // namespace

void hif8_fused_mm_sm86(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight_u8,
    c10::optional<torch::Tensor> const& bias,
    bool enable_input_qdq,
    bool enable_output_requant,
    std::string const& compute_dtype) {
  CHECK_INPUT(out);
  CHECK_INPUT(input);
  CHECK_INPUT(weight_u8);

  TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
  TORCH_CHECK(weight_u8.dim() == 2, "weight_u8 must be [K, N]");
  TORCH_CHECK(out.dim() == 2, "out must be [M, N]");
  TORCH_CHECK(weight_u8.scalar_type() == at::ScalarType::Byte, "weight_u8 must be uint8");

  const auto m = input.size(0);
  const auto k = input.size(1);
  const auto wk = weight_u8.size(0);
  const auto n = weight_u8.size(1);
  TORCH_CHECK(k == wk, "shape mismatch: input K != weight K");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "out shape must be [M, N]");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getDefaultCUDAStream(input.device().index()).stream();

  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  const c10::Half* bias_ptr_h = nullptr;
  const c10::BFloat16* bias_ptr_bf = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D [N]");
    TORCH_CHECK(bias.value().size(0) == n, "bias size mismatch with N");
  }

  // compute_dtype is kept for interface compatibility and future specialization.
  (void)compute_dtype;

  if (out.scalar_type() == at::ScalarType::Half) {
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "input dtype must match out dtype");
    if (bias.has_value()) {
      TORCH_CHECK(bias.value().scalar_type() == at::ScalarType::Half, "bias dtype must match out dtype");
      bias_ptr_h = reinterpret_cast<const c10::Half*>(bias.value().data_ptr<c10::Half>());
    }
    hif8_fused_mm_kernel<c10::Half><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Half>(),
        input.data_ptr<c10::Half>(),
        weight_u8.data_ptr<uint8_t>(),
        bias_ptr_h,
        m,
        n,
        k,
        enable_input_qdq,
        enable_output_requant);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "input dtype must match out dtype");
    if (bias.has_value()) {
      TORCH_CHECK(bias.value().scalar_type() == at::ScalarType::BFloat16, "bias dtype must match out dtype");
      bias_ptr_bf = reinterpret_cast<const c10::BFloat16*>(bias.value().data_ptr<c10::BFloat16>());
    }
    hif8_fused_mm_kernel<c10::BFloat16><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::BFloat16>(),
        input.data_ptr<c10::BFloat16>(),
        weight_u8.data_ptr<uint8_t>(),
        bias_ptr_bf,
        m,
        n,
        k,
        enable_input_qdq,
        enable_output_requant);
  } else {
    TORCH_CHECK(false, "out dtype must be fp16 or bf16");
  }

  CHECK_CUDA_SUCCESS(cudaGetLastError());
}
