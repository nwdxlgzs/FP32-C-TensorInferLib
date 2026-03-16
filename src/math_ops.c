#include "tensor.h"
#include "math_ops.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ==================== 一元运算操作函数 ==================== */

static float neg_op(float x) { return -x; }
static float abs_op(float x) { return fabsf(x); }
static float sqrt_op(float x) { return sqrtf(x); }
static float rsqrt_op(float x) { return 1.0f / sqrtf(x); }
static float exp_op(float x) { return expf(x); }
static float log_op(float x) { return logf(x); }
static float log1p_op(float x) { return log1pf(x); }
static float sin_op(float x) { return sinf(x); }
static float cos_op(float x) { return cosf(x); }
static float tan_op(float x) { return tanf(x); }
static float asin_op(float x) { return asinf(x); }
static float acos_op(float x) { return acosf(x); }
static float atan_op(float x) { return atanf(x); }
static float sinh_op(float x) { return sinhf(x); }
static float cosh_op(float x) { return coshf(x); }
static float tanh_op(float x) { return tanhf(x); }
static float asinh_op(float x) { return asinhf(x); }
static float acosh_op(float x) { return acoshf(x); }
static float atanh_op(float x) { return atanhf(x); }
static float ceil_op(float x) { return ceilf(x); }
static float floor_op(float x) { return floorf(x); }
static float round_op(float x) { return roundf(x); }
static float trunc_op(float x) { return truncf(x); }
static float sign_op(float x) { return (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f); }
static float square_op(float x) { return x * x; }
static float cube_op(float x) { return x * x * x; }
static float reciprocal_op(float x) { return 1.0f / x; }
static float erf_op(float x) { return erff(x); }
static float erfc_op(float x) { return erfcf(x); }

/* ==================== 一元运算 API ==================== */

TensorStatus tensor_neg(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, neg_op); }
TensorStatus tensor_abs(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, abs_op); }
TensorStatus tensor_sqrt(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, sqrt_op); }
TensorStatus tensor_rsqrt(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, rsqrt_op); }
TensorStatus tensor_exp(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, exp_op); }
TensorStatus tensor_log(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, log_op); }
TensorStatus tensor_log1p(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, log1p_op); }
TensorStatus tensor_sin(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, sin_op); }
TensorStatus tensor_cos(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, cos_op); }
TensorStatus tensor_tan(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, tan_op); }
TensorStatus tensor_asin(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, asin_op); }
TensorStatus tensor_acos(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, acos_op); }
TensorStatus tensor_atan(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, atan_op); }
TensorStatus tensor_sinh(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, sinh_op); }
TensorStatus tensor_cosh(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, cosh_op); }
TensorStatus tensor_tanh(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, tanh_op); }
TensorStatus tensor_asinh(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, asinh_op); }
TensorStatus tensor_acosh(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, acosh_op); }
TensorStatus tensor_atanh(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, atanh_op); }
TensorStatus tensor_ceil(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, ceil_op); }
TensorStatus tensor_floor(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, floor_op); }
TensorStatus tensor_round(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, round_op); }
TensorStatus tensor_trunc(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, trunc_op); }
TensorStatus tensor_sign(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, sign_op); }
TensorStatus tensor_square(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, square_op); }
TensorStatus tensor_cube(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, cube_op); }
TensorStatus tensor_reciprocal(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, reciprocal_op); }
TensorStatus tensor_erf(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, erf_op); }
TensorStatus tensor_erfc(const Tensor *x, Tensor *out) { return util_unary_op_general(x, out, erfc_op); }

/* ==================== 二元运算操作函数 ==================== */

static float add_op(float a, float b) { return a + b; }
static float sub_op(float a, float b) { return a - b; }
static float mul_op(float a, float b) { return a * b; }
static float div_op(float a, float b) { return a / b; }
static float pow_op(float a, float b) { return powf(a, b); }
static float max_op(float a, float b) { return fmaxf(a, b); }
static float min_op(float a, float b) { return fminf(a, b); }
static float fmod_op(float a, float b) { return fmodf(a, b); }
static float hypot_op(float a, float b) { return hypotf(a, b); }
static float atan2_op(float a, float b) { return atan2f(a, b); }
static float copysign_op(float a, float b) { return copysignf(a, b); }
static float nextafter_op(float a, float b) { return nextafterf(a, b); }

/* ==================== 二元运算 API ==================== */

TensorStatus tensor_add(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, add_op);
}
TensorStatus tensor_sub(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, sub_op);
}
TensorStatus tensor_mul(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, mul_op);
}
TensorStatus tensor_div(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, div_op);
}
TensorStatus tensor_pow(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, pow_op);
}
TensorStatus tensor_maximum(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, max_op);
}
TensorStatus tensor_minimum(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, min_op);
}
TensorStatus tensor_fmod(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, fmod_op);
}
TensorStatus tensor_hypot(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, hypot_op);
}
TensorStatus tensor_atan2(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, atan2_op);
}
TensorStatus tensor_copysign(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, copysign_op);
}
TensorStatus tensor_nextafter(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, nextafter_op);
}

/* ==================== 标量二元运算 API ==================== */

TensorStatus tensor_add_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar(a, scalar, out, add_op);
}
TensorStatus tensor_sub_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar(a, scalar, out, sub_op);
}
TensorStatus tensor_mul_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar(a, scalar, out, mul_op);
}
TensorStatus tensor_div_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar(a, scalar, out, div_op);
}
TensorStatus tensor_pow_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar(a, scalar, out, pow_op);
}
TensorStatus tensor_maximum_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar(a, scalar, out, max_op);
}
TensorStatus tensor_minimum_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar(a, scalar, out, min_op);
}

/* ==================== 三元运算操作函数 ==================== */

static float clamp_op(float x, float lo, float hi)
{
    if (x < lo)
        return lo;
    if (x > hi)
        return hi;
    return x;
}

static float where_op(float cond, float x, float y)
{
    return cond != 0.0f ? x : y;
}

/* ==================== 三元运算 API ==================== */

TensorStatus tensor_clamp(const Tensor *x, const Tensor *min,
                          const Tensor *max, Tensor *out)
{
    return util_ternary_op_general(x, min, max, out, clamp_op);
}

TensorStatus tensor_clamp_scalar(const Tensor *x, float min_val,
                                 float max_val, Tensor *out)
{
    /* 将标量包装为临时张量后调用三元运算 */
    Tensor min_tensor, max_tensor;
    min_tensor.data = &min_val;
    min_tensor.ndim = 0;
    min_tensor.dims = NULL;
    min_tensor.strides = NULL;
    min_tensor.size = 1;
    min_tensor.ref_count = NULL;
    min_tensor.owns_dims_strides = 0;

    max_tensor.data = &max_val;
    max_tensor.ndim = 0;
    max_tensor.dims = NULL;
    max_tensor.strides = NULL;
    max_tensor.size = 1;
    max_tensor.ref_count = NULL;
    max_tensor.owns_dims_strides = 0;

    return util_ternary_op_general(x, &min_tensor, &max_tensor, out, clamp_op);
}

TensorStatus tensor_where(const Tensor *condition, const Tensor *x,
                          const Tensor *y, Tensor *out)
{
    return util_ternary_op_general(condition, x, y, out, where_op);
}