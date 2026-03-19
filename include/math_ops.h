#ifndef TENSOR_MATH_OPS_H
#define TENSOR_MATH_OPS_H

#include "tensor.h"

/**
 * @file math_ops.h
 * @brief 数学运算：一元、二元、三元操作（朴素循环实现，支持广播）
 */

#ifdef __cplusplus
extern "C"
{
#endif

    /* ==================== 一元运算 ==================== */

    /**
     * @brief 逐元素求绝对值: out = |x|
     * @param x 输入张量
     * @param out 输出张量（与 x 形状相同）
     * @return TensorStatus
     */
    TensorStatus tensor_abs(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素反余弦: out = acos(x)
     * @param x 输入张量（值域 [-1, 1]）
     * @param out 输出张量（弧度制）
     */
    TensorStatus tensor_acos(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素反双曲余弦: out = acosh(x)
     * @param x 输入张量（x >= 1）
     * @param out 输出张量
     */
    TensorStatus tensor_acosh(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素反正弦: out = asin(x)
     * @param x 输入张量（值域 [-1, 1]）
     * @param out 输出张量（弧度制）
     */
    TensorStatus tensor_asin(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素反双曲正弦: out = asinh(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_asinh(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素反正切: out = atan(x)
     * @param x 输入张量
     * @param out 输出张量（弧度制）
     */
    TensorStatus tensor_atan(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素反双曲正切: out = atanh(x)
     * @param x 输入张量（值域 (-1, 1)）
     * @param out 输出张量
     */
    TensorStatus tensor_atanh(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素向上取整: out = ceil(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_ceil(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素余弦: out = cos(x)
     * @param x 输入张量（弧度制）
     * @param out 输出张量
     */
    TensorStatus tensor_cos(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素双曲余弦: out = cosh(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_cosh(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素立方: out = x^3
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_cube(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素误差函数: out = erf(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_erf(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素互补误差函数: out = erfc(x) = 1 - erf(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_erfc(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素指数: out = exp(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_exp(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素向下取整: out = floor(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_floor(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素自然对数: out = log(x) （x > 0）
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_log(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素 log(1+x)，对接近0的值精度更高
     * @param x 输入张量（x > -1）
     * @param out 输出张量
     */
    TensorStatus tensor_log1p(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素求负: out = -x
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_neg(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素倒数: out = 1/x
     * @param x 输入张量（x != 0）
     * @param out 输出张量
     */
    TensorStatus tensor_reciprocal(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素四舍五入到最近整数
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_round(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素平方根的倒数: out = 1/sqrt(x)
     * @param x 输入张量（x > 0）
     * @param out 输出张量
     */
    TensorStatus tensor_rsqrt(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素符号函数: out = -1, 0, 1
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_sign(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素正弦: out = sin(x)
     * @param x 输入张量（弧度制）
     * @param out 输出张量
     */
    TensorStatus tensor_sin(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素双曲正弦: out = sinh(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_sinh(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素平方根: out = sqrt(x)
     * @param x 输入张量（x >= 0）
     * @param out 输出张量
     */
    TensorStatus tensor_sqrt(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素平方: out = x^2
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_square(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素正切: out = tan(x)
     * @param x 输入张量（弧度制）
     * @param out 输出张量
     */
    TensorStatus tensor_tan(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素双曲正切: out = tanh(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_tanh(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素截断小数部分: out = trunc(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_trunc(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素 sigmoid 函数: out = 1 / (1 + exp(-x))
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_sigmoid(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素 logit 函数: out = log(p / (1-p))
     * @param p 输入张量（概率，应在 (0,1) 范围内）
     * @param eps 剪裁阈值，p 被剪裁到 [eps, 1-eps] 以防止对数溢出
     * @param out 输出张量
     */
    TensorStatus tensor_logit(const Tensor *p, float eps, Tensor *out);

    /**
     * @brief 逐元素 gamma 函数（阶乘的推广）
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_gamma(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素 log-gamma 函数
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_lgamma(const Tensor *x, Tensor *out);

    /**
     * @brief 逐元素第一类修正贝塞尔函数 I0(x)
     * @param x 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_bessel_i0(const Tensor *x, Tensor *out);

    /* ==================== 二元运算（张量-张量，支持广播） ==================== */

    /**
     * @brief 逐元素加法: out = a + b
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量（形状必须与广播结果一致）
     */
    TensorStatus tensor_add(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素 atan2: out = atan2(a, b)
     * @param a 输入张量（y 坐标）
     * @param b 输入张量（x 坐标）
     * @param out 输出张量（弧度制，值域 [-π, π]）
     */
    TensorStatus tensor_atan2(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素复制符号: out = copysign(a, b) （结果绝对值来自 a，符号来自 b）
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_copysign(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素除法: out = a / b
     * @param a 输入张量
     * @param b 输入张量（b 中不应有 0）
     * @param out 输出张量
     */
    TensorStatus tensor_div(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素取模（浮点数余数）: out = a % b （符号与 a 一致）
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_fmod(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素计算直角三角形斜边: out = sqrt(a^2 + b^2)
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_hypot(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素取最大值: out = max(a, b)
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_maximum(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素取最小值: out = min(a, b)
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_minimum(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素乘法: out = a * b
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_mul(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素 nextafter: out = nextafter(a, b) （向 b 方向的下一个可表示浮点数）
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_nextafter(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素幂运算: out = a^b
     * @param a 输入张量（底数）
     * @param b 输入张量（指数）
     * @param out 输出张量
     */
    TensorStatus tensor_pow(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素减法: out = a - b
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_sub(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素近似相等: out = isclose(a, b)  (rtol, atol 通过 user_data 传递)
     * @param a 输入张量
     * @param b 输入张量
     * @param rtol 相对容忍度
     * @param atol 绝对容忍度
     * @param out 输出布尔张量
     */
    TensorStatus tensor_isclose(const Tensor *a, const Tensor *b, float rtol, float atol, Tensor *out);

    /**
     * @brief 逐元素 IEEE 余数: out = remainder(a, b)
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     */
    TensorStatus tensor_remainder(const Tensor *a, const Tensor *b, Tensor *out);

    /* ==================== 标量二元运算（张量-标量） ==================== */

    /**
     * @brief 逐元素加标量: out = a + scalar
     * @param a 输入张量
     * @param scalar 标量值
     * @param out 输出张量
     */
    TensorStatus tensor_add_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素除标量: out = a / scalar
     * @param a 输入张量
     * @param scalar 标量值（非零）
     * @param out 输出张量
     */
    TensorStatus tensor_div_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素取最大值（与标量）: out = max(a, scalar)
     * @param a 输入张量
     * @param scalar 标量值
     * @param out 输出张量
     */
    TensorStatus tensor_maximum_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素取最小值（与标量）: out = min(a, scalar)
     * @param a 输入张量
     * @param scalar 标量值
     * @param out 输出张量
     */
    TensorStatus tensor_minimum_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素乘标量: out = a * scalar
     * @param a 输入张量
     * @param scalar 标量值
     * @param out 输出张量
     */
    TensorStatus tensor_mul_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素幂标量: out = a^scalar
     * @param a 输入张量
     * @param scalar 标量指数
     * @param out 输出张量
     */
    TensorStatus tensor_pow_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素减标量: out = a - scalar
     * @param a 输入张量
     * @param scalar 标量值
     * @param out 输出张量
     */
    TensorStatus tensor_sub_scalar(const Tensor *a, float scalar, Tensor *out);

    /* ==================== 三元运算（支持广播） ==================== */

    /**
     * @brief 逐元素裁剪: out = min(max(x, min), max)
     * @param x 输入张量
     * @param min 下界张量，支持广播
     * @param max 上界张量，支持广播
     * @param out 输出张量
     */
    TensorStatus tensor_clamp(const Tensor *x, const Tensor *min, const Tensor *max, Tensor *out);

    /**
     * @brief 逐元素裁剪（标量版本）: out = min(max(x, min_val), max_val)
     * @param x 输入张量
     * @param min_val 下界标量
     * @param max_val 上界标量
     * @param out 输出张量
     */
    TensorStatus tensor_clamp_scalar(const Tensor *x, float min_val, float max_val, Tensor *out);

    /**
     * @brief 条件选择: out = condition ? x : y
     * @param condition 布尔张量（0.0 或 1.0），支持广播
     * @param x 输入张量，支持广播
     * @param y 输入张量，支持广播
     * @param out 输出张量
     */
    TensorStatus tensor_where(const Tensor *condition, const Tensor *x, const Tensor *y, Tensor *out);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_MATH_OPS_H