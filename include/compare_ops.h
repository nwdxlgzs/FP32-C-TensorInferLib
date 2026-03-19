#ifndef TENSOR_COMPARE_OPS_H
#define TENSOR_COMPARE_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file compare_ops.h
     * @brief 比较与逻辑运算（返回布尔张量，值为 0.0 或 1.0）
     */

    /* ==================== 二元比较 ==================== */

    /**
     * @brief 逐元素比较相等: out = (a == b)
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出布尔张量（形状同广播后结果）
     * @return TensorStatus
     */
    TensorStatus tensor_equal(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素比较不等: out = (a != b)
     */
    TensorStatus tensor_not_equal(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素比较小于: out = (a < b)
     */
    TensorStatus tensor_less(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素比较小于等于: out = (a <= b)
     */
    TensorStatus tensor_less_equal(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素比较大于: out = (a > b)
     */
    TensorStatus tensor_greater(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素比较大于等于: out = (a >= b)
     */
    TensorStatus tensor_greater_equal(const Tensor *a, const Tensor *b, Tensor *out);

    /* ==================== 标量比较 ==================== */

    /**
     * @brief 逐元素与标量比较相等: out = (a == scalar)
     */
    TensorStatus tensor_equal_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素与标量比较不等: out = (a != scalar)
     */
    TensorStatus tensor_not_equal_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素与标量比较小于: out = (a < scalar)
     */
    TensorStatus tensor_less_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素与标量比较小于等于: out = (a <= scalar)
     */
    TensorStatus tensor_less_equal_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素与标量比较大于: out = (a > scalar)
     */
    TensorStatus tensor_greater_scalar(const Tensor *a, float scalar, Tensor *out);

    /**
     * @brief 逐元素与标量比较大于等于: out = (a >= scalar)
     */
    TensorStatus tensor_greater_equal_scalar(const Tensor *a, float scalar, Tensor *out);

    /* ==================== 逻辑运算 ==================== */

    /**
     * @brief 逐元素逻辑与: out = a && b
     * @param a 输入布尔张量（非零视为真）
     * @param b 输入布尔张量
     * @param out 输出布尔张量
     */
    TensorStatus tensor_logical_and(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素逻辑或: out = a || b
     */
    TensorStatus tensor_logical_or(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素逻辑异或: out = a ^ b
     */
    TensorStatus tensor_logical_xor(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逐元素逻辑非: out = !a
     * @param a 输入布尔张量
     * @param out 输出布尔张量
     */
    TensorStatus tensor_logical_not(const Tensor *a, Tensor *out);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_COMPARE_OPS_H