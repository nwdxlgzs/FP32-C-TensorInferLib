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

    TensorStatus tensor_equal(const Tensor *a, const Tensor *b, Tensor *out);
    TensorStatus tensor_not_equal(const Tensor *a, const Tensor *b, Tensor *out);
    TensorStatus tensor_less(const Tensor *a, const Tensor *b, Tensor *out);
    TensorStatus tensor_less_equal(const Tensor *a, const Tensor *b, Tensor *out);
    TensorStatus tensor_greater(const Tensor *a, const Tensor *b, Tensor *out);
    TensorStatus tensor_greater_equal(const Tensor *a, const Tensor *b, Tensor *out);

    /* ==================== 标量比较 ==================== */

    TensorStatus tensor_equal_scalar(const Tensor *a, float scalar, Tensor *out);
    TensorStatus tensor_not_equal_scalar(const Tensor *a, float scalar, Tensor *out);
    TensorStatus tensor_less_scalar(const Tensor *a, float scalar, Tensor *out);
    TensorStatus tensor_less_equal_scalar(const Tensor *a, float scalar, Tensor *out);
    TensorStatus tensor_greater_scalar(const Tensor *a, float scalar, Tensor *out);
    TensorStatus tensor_greater_equal_scalar(const Tensor *a, float scalar, Tensor *out);

    /* ==================== 逻辑运算 ==================== */

    /**
     * @brief 逻辑与: a && b（a,b 应为布尔，非零视为真）
     */
    TensorStatus tensor_logical_and(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逻辑或: a || b
     */
    TensorStatus tensor_logical_or(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逻辑异或: a ^ b
     */
    TensorStatus tensor_logical_xor(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 逻辑非: !a
     */
    TensorStatus tensor_logical_not(const Tensor *a, Tensor *out);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_COMPARE_OPS_H