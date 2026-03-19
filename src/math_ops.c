#include "tensor.h"
#include "math_ops.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @file math_ops.c
 * @brief 数学运算的实现：一元、二元、三元操作（朴素循环实现，支持广播）
 *
 * 包含逐元素的数学函数（绝对值、三角函数、指数对数、取整、符号、平方立方等）、
 * 二元算术运算（加、减、乘、除、幂、比较等）、标量版本、三元运算（裁剪、条件选择）
 * 以及一些特殊函数（sigmoid, logit, gamma, bessel_i0 等）。
 * 所有运算均通过通用迭代器 util_unary_op_general, util_binary_op_general,
 * util_ternary_op_general 实现，支持广播和写时拷贝。
 */

/* ==================== 一元运算操作函数（静态） ==================== */

static float neg_op(float x, void *user_data) { return -x; }
static float abs_op(float x, void *user_data) { return fabsf(x); }
static float sqrt_op(float x, void *user_data) { return sqrtf(x); }
static float rsqrt_op(float x, void *user_data) { return 1.0f / sqrtf(x); }
static float exp_op(float x, void *user_data) { return expf(x); }
static float log_op(float x, void *user_data) { return logf(x); }
static float log1p_op(float x, void *user_data) { return log1pf(x); }
static float sin_op(float x, void *user_data) { return sinf(x); }
static float cos_op(float x, void *user_data) { return cosf(x); }
static float tan_op(float x, void *user_data) { return tanf(x); }
static float asin_op(float x, void *user_data) { return asinf(x); }
static float acos_op(float x, void *user_data) { return acosf(x); }
static float atan_op(float x, void *user_data) { return atanf(x); }
static float sinh_op(float x, void *user_data) { return sinhf(x); }
static float cosh_op(float x, void *user_data) { return coshf(x); }
static float tanh_op(float x, void *user_data) { return tanhf(x); }
static float asinh_op(float x, void *user_data) { return asinhf(x); }
static float acosh_op(float x, void *user_data) { return acoshf(x); }
static float atanh_op(float x, void *user_data) { return atanhf(x); }
static float ceil_op(float x, void *user_data) { return ceilf(x); }
static float floor_op(float x, void *user_data) { return floorf(x); }
static float round_op(float x, void *user_data) { return roundf(x); }
static float trunc_op(float x, void *user_data) { return truncf(x); }
static float sign_op(float x, void *user_data) { return (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f); }
static float square_op(float x, void *user_data) { return x * x; }
static float cube_op(float x, void *user_data) { return x * x * x; }
static float reciprocal_op(float x, void *user_data) { return 1.0f / x; }
static float erf_op(float x, void *user_data) { return erff(x); }
static float erfc_op(float x, void *user_data) { return erfcf(x); }

/* ==================== 一元运算 API ==================== */

TensorStatus tensor_neg(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, neg_op); }
TensorStatus tensor_abs(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, abs_op); }
TensorStatus tensor_sqrt(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, sqrt_op); }
TensorStatus tensor_rsqrt(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, rsqrt_op); }
TensorStatus tensor_exp(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, exp_op); }
TensorStatus tensor_log(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, log_op); }
TensorStatus tensor_log1p(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, log1p_op); }
TensorStatus tensor_sin(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, sin_op); }
TensorStatus tensor_cos(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, cos_op); }
TensorStatus tensor_tan(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, tan_op); }
TensorStatus tensor_asin(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, asin_op); }
TensorStatus tensor_acos(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, acos_op); }
TensorStatus tensor_atan(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, atan_op); }
TensorStatus tensor_sinh(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, sinh_op); }
TensorStatus tensor_cosh(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, cosh_op); }
TensorStatus tensor_tanh(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, tanh_op); }
TensorStatus tensor_asinh(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, asinh_op); }
TensorStatus tensor_acosh(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, acosh_op); }
TensorStatus tensor_atanh(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, atanh_op); }
TensorStatus tensor_ceil(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, ceil_op); }
TensorStatus tensor_floor(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, floor_op); }
TensorStatus tensor_round(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, round_op); }
TensorStatus tensor_trunc(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, trunc_op); }
TensorStatus tensor_sign(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, sign_op); }
TensorStatus tensor_square(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, square_op); }
TensorStatus tensor_cube(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, cube_op); }
TensorStatus tensor_reciprocal(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, reciprocal_op); }
TensorStatus tensor_erf(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, erf_op); }
TensorStatus tensor_erfc(const Tensor *x, Tensor *out) { return util_unary_op_general_NUD(x, out, erfc_op); }

/* ==================== 二元运算操作函数（静态） ==================== */

static float add_op(float a, float b, void *user_data) { return a + b; }
static float sub_op(float a, float b, void *user_data) { return a - b; }
static float mul_op(float a, float b, void *user_data) { return a * b; }
static float div_op(float a, float b, void *user_data) { return a / b; }
static float pow_op(float a, float b, void *user_data) { return powf(a, b); }
static float max_op(float a, float b, void *user_data) { return fmaxf(a, b); }
static float min_op(float a, float b, void *user_data) { return fminf(a, b); }
static float fmod_op(float a, float b, void *user_data) { return fmodf(a, b); }
static float hypot_op(float a, float b, void *user_data) { return hypotf(a, b); }
static float atan2_op(float a, float b, void *user_data) { return atan2f(a, b); }
static float copysign_op(float a, float b, void *user_data) { return copysignf(a, b); }
static float nextafter_op(float a, float b, void *user_data) { return nextafterf(a, b); }

typedef struct
{
    float rtol;
    float atol;
} isclose_params;

static float isclose_op(float a, float b, void *user_data)
{
    isclose_params *p = (isclose_params *)user_data;
    float diff = fabsf(a - b);
    float tol = p->atol + p->rtol * fabsf(b);
    return (diff <= tol) ? 1.0f : 0.0f;
}

static float remainder_op(float a, float b, void *user_data)
{
    (void)user_data;
    return remainderf(a, b);
}

/* ==================== 二元运算 API ==================== */

TensorStatus tensor_isclose(const Tensor *a, const Tensor *b, float rtol, float atol, Tensor *out)
{
    if (!a || !b || !out)
        return TENSOR_ERR_NULL_PTR;
    isclose_params params = {rtol, atol};
    return util_binary_op_general(a, b, out, isclose_op, &params);
}

TensorStatus tensor_remainder(const Tensor *a, const Tensor *b, Tensor *out)
{
    if (!a || !b || !out)
        return TENSOR_ERR_NULL_PTR;
    return util_binary_op_general_NUD(a, b, out, remainder_op);
}
TensorStatus tensor_add(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, add_op);
}
TensorStatus tensor_sub(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, sub_op);
}
TensorStatus tensor_mul(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, mul_op);
}
TensorStatus tensor_div(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, div_op);
}
TensorStatus tensor_pow(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, pow_op);
}
TensorStatus tensor_maximum(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, max_op);
}
TensorStatus tensor_minimum(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, min_op);
}
TensorStatus tensor_fmod(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, fmod_op);
}
TensorStatus tensor_hypot(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, hypot_op);
}
TensorStatus tensor_atan2(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, atan2_op);
}
TensorStatus tensor_copysign(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, copysign_op);
}
TensorStatus tensor_nextafter(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, nextafter_op);
}

/* ==================== 标量二元运算 API ==================== */

TensorStatus tensor_add_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar_NUD(a, scalar, out, add_op);
}
TensorStatus tensor_sub_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar_NUD(a, scalar, out, sub_op);
}
TensorStatus tensor_mul_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar_NUD(a, scalar, out, mul_op);
}
TensorStatus tensor_div_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar_NUD(a, scalar, out, div_op);
}
TensorStatus tensor_pow_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar_NUD(a, scalar, out, pow_op);
}
TensorStatus tensor_maximum_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar_NUD(a, scalar, out, max_op);
}
TensorStatus tensor_minimum_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return util_binary_op_scalar_NUD(a, scalar, out, min_op);
}

/* ==================== 三元运算操作函数（静态） ==================== */

/**
 * @brief 裁剪操作：将 x 限制在 [lo, hi] 区间内
 */
static float clamp_op(float x, float lo, float hi, void *user_data)
{
    if (x < lo)
        return lo;
    if (x > hi)
        return hi;
    return x;
}

/**
 * @brief 条件选择操作：根据 cond 选择 x（cond != 0）或 y
 */
static float where_op(float cond, float x, float y, void *user_data)
{
    return cond != 0.0f ? x : y;
}

/* ==================== 三元运算 API ==================== */

TensorStatus tensor_clamp(const Tensor *x, const Tensor *min,
                          const Tensor *max, Tensor *out)
{
    return util_ternary_op_general_NUD(x, min, max, out, clamp_op);
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

    return util_ternary_op_general_NUD(x, &min_tensor, &max_tensor, out, clamp_op);
}

TensorStatus tensor_where(const Tensor *condition, const Tensor *x,
                          const Tensor *y, Tensor *out)
{
    return util_ternary_op_general_NUD(condition, x, y, out, where_op);
}

/* ==================== 特殊函数操作（静态及API） ==================== */

/**
 * @brief sigmoid 函数: 1 / (1 + exp(-x))
 */
static float sigmoid_op(float x, void *user_data)
{
    return 1.0f / (1.0f + expf(-x));
}

TensorStatus tensor_sigmoid(const Tensor *x, Tensor *out)
{
    return util_unary_op_general_NUD(x, out, sigmoid_op);
}

/**
 * @brief logit 函数（带剪裁）: log(p / (1-p))
 * @param p 输入概率
 * @param user_data 指向 eps 的指针，用于剪裁 p 到 [eps, 1-eps]
 */
static float logit_op_with_eps(float p, void *user_data)
{
    float eps = *(float *)user_data;
    float clipped = fminf(fmaxf(p, eps), 1.0f - eps);
    return logf(clipped / (1.0f - clipped));
}

TensorStatus tensor_logit(const Tensor *p, float eps, Tensor *out)
{
    if (!p || !out)
        return TENSOR_ERR_NULL_PTR;
    if (eps <= 0.0f || eps >= 0.5f)
        return TENSOR_ERR_INVALID_PARAM; // eps 必须在 (0, 0.5) 区间
    return util_unary_op_general(p, out, logit_op_with_eps, &eps);
}

/**
 * @brief gamma 函数
 */
static float gamma_op(float x, void *user_data)
{
    return tgammaf(x); // C11 标准函数
}

TensorStatus tensor_gamma(const Tensor *x, Tensor *out)
{
    return util_unary_op_general_NUD(x, out, gamma_op);
}

/**
 * @brief log-gamma 函数
 */
static float lgamma_op(float x, void *user_data)
{
    return lgammaf(x);
}

TensorStatus tensor_lgamma(const Tensor *x, Tensor *out)
{
    return util_unary_op_general_NUD(x, out, lgamma_op);
}

/**
 * @brief 第一类修正贝塞尔函数 I0(x) 的数值实现（使用多项式近似）
 */
static float bessel_i0_op(float x, void *user_data)
{
    float ax = fabsf(x);
    if (ax < 3.75f)
    {
        // 多项式近似（源自 Abramowitz and Stegun）
        float y = x / 3.75f;
        y = y * y;
        return 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f +
                                                               y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
    }
    else
    {
        float y = 3.75f / ax;
        float ans = expf(ax) / sqrtf(ax) * (0.39894228f + y * (0.01328592f + y * (0.00225319f + y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f + y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f))))))));
        return ans;
    }
}

TensorStatus tensor_bessel_i0(const Tensor *x, Tensor *out)
{
    return util_unary_op_general_NUD(x, out, bessel_i0_op);
}