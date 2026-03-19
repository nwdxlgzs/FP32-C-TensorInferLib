#include "tensor.h"
#include "compare_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/**
 * @file compare_ops.c
 * @brief 比较与逻辑运算的实现（返回布尔张量，值为 0.0 或 1.0）
 *
 * 本文件包含逐元素比较运算（等于、不等于、小于、小于等于、大于、大于等于）
 * 以及与标量的比较运算，还有逻辑运算（与、或、异或、非）的实现。
 * 所有运算均通过通用二元/一元迭代器 util_binary_op_general_NUD 和
 * util_unary_op_general_NUD 完成，支持广播。
 */

/* ==================== 比较操作函数（静态） ==================== */

/**
 * @brief 相等比较操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a == b 时返回 1.0f，否则返回 0.0f
 */
static float eq_op(float a, float b, void *user_data) { return (a == b) ? 1.0f : 0.0f; }

/**
 * @brief 不等比较操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a != b 时返回 1.0f，否则返回 0.0f
 */
static float ne_op(float a, float b, void *user_data) { return (a != b) ? 1.0f : 0.0f; }

/**
 * @brief 小于比较操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a < b 时返回 1.0f，否则返回 0.0f
 */
static float lt_op(float a, float b, void *user_data) { return (a < b) ? 1.0f : 0.0f; }

/**
 * @brief 小于等于比较操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a <= b 时返回 1.0f，否则返回 0.0f
 */
static float le_op(float a, float b, void *user_data) { return (a <= b) ? 1.0f : 0.0f; }

/**
 * @brief 大于比较操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a > b 时返回 1.0f，否则返回 0.0f
 */
static float gt_op(float a, float b, void *user_data) { return (a > b) ? 1.0f : 0.0f; }

/**
 * @brief 大于等于比较操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a >= b 时返回 1.0f，否则返回 0.0f
 */
static float ge_op(float a, float b, void *user_data) { return (a >= b) ? 1.0f : 0.0f; }

/* ==================== 二元比较 API ==================== */

/**
 * @brief 逐元素比较相等: out = (a == b)
 * @param a 输入张量
 * @param b 输入张量
 * @param out 输出布尔张量（形状同广播后结果）
 * @return TensorStatus
 */
TensorStatus tensor_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, eq_op);
}

/**
 * @brief 逐元素比较不等: out = (a != b)
 * @param a 输入张量
 * @param b 输入张量
 * @param out 输出布尔张量（形状同广播后结果）
 * @return TensorStatus
 */
TensorStatus tensor_not_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, ne_op);
}

/**
 * @brief 逐元素比较小于: out = (a < b)
 * @param a 输入张量
 * @param b 输入张量
 * @param out 输出布尔张量（形状同广播后结果）
 * @return TensorStatus
 */
TensorStatus tensor_less(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, lt_op);
}

/**
 * @brief 逐元素比较小于等于: out = (a <= b)
 * @param a 输入张量
 * @param b 输入张量
 * @param out 输出布尔张量（形状同广播后结果）
 * @return TensorStatus
 */
TensorStatus tensor_less_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, le_op);
}

/**
 * @brief 逐元素比较大于: out = (a > b)
 * @param a 输入张量
 * @param b 输入张量
 * @param out 输出布尔张量（形状同广播后结果）
 * @return TensorStatus
 */
TensorStatus tensor_greater(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, gt_op);
}

/**
 * @brief 逐元素比较大于等于: out = (a >= b)
 * @param a 输入张量
 * @param b 输入张量
 * @param out 输出布尔张量（形状同广播后结果）
 * @return TensorStatus
 */
TensorStatus tensor_greater_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, ge_op);
}

/* ==================== 标量比较 API ==================== */

/**
 * @brief 通用标量比较操作辅助函数
 * @param a 输入张量
 * @param scalar 标量值
 * @param out 输出张量
 * @param op 二元操作函数（比较操作）
 * @return TensorStatus
 */
static TensorStatus scalar_compare_op(const Tensor *a, float scalar,
                                      Tensor *out, BinaryOp op)
{
    Tensor scalar_tensor;
    scalar_tensor.data = &scalar;
    scalar_tensor.ndim = 0;
    scalar_tensor.dims = NULL;
    scalar_tensor.strides = NULL;
    scalar_tensor.size = 1;
    scalar_tensor.ref_count = NULL;
    scalar_tensor.owns_dims_strides = 0;
    return util_binary_op_general_NUD(a, &scalar_tensor, out, op);
}

/**
 * @brief 逐元素与标量比较相等: out = (a == scalar)
 * @param a 输入张量
 * @param scalar 标量值
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, eq_op);
}

/**
 * @brief 逐元素与标量比较不等: out = (a != scalar)
 * @param a 输入张量
 * @param scalar 标量值
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_not_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, ne_op);
}

/**
 * @brief 逐元素与标量比较小于: out = (a < scalar)
 * @param a 输入张量
 * @param scalar 标量值
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_less_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, lt_op);
}

/**
 * @brief 逐元素与标量比较小于等于: out = (a <= scalar)
 * @param a 输入张量
 * @param scalar 标量值
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_less_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, le_op);
}

/**
 * @brief 逐元素与标量比较大于: out = (a > scalar)
 * @param a 输入张量
 * @param scalar 标量值
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_greater_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, gt_op);
}

/**
 * @brief 逐元素与标量比较大于等于: out = (a >= scalar)
 * @param a 输入张量
 * @param scalar 标量值
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_greater_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, ge_op);
}

/* ==================== 逻辑操作函数（静态） ==================== */

/**
 * @brief 逻辑与操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a 和 b 均非零时返回 1.0f，否则返回 0.0f
 */
static float and_op(float a, float b, void *user_data) { return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f; }

/**
 * @brief 逻辑或操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a 或 b 非零时返回 1.0f，否则返回 0.0f
 */
static float or_op(float a, float b, void *user_data) { return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f; }

/**
 * @brief 逻辑异或操作
 * @param a 左操作数
 * @param b 右操作数
 * @param user_data 未使用
 * @return a 和 b 的逻辑值不同时返回 1.0f，否则返回 0.0f
 */
static float xor_op(float a, float b, void *user_data) { return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f; }

/**
 * @brief 逻辑非操作（一元）
 * @param a 操作数
 * @param user_data 未使用
 * @return a 为 0.0f 时返回 1.0f，否则返回 0.0f
 */
static float not_op(float a, void *user_data) { return (a == 0.0f) ? 1.0f : 0.0f; }

/* ==================== 逻辑运算 API ==================== */

/**
 * @brief 逐元素逻辑与: out = a && b
 * @param a 输入布尔张量（非零视为真）
 * @param b 输入布尔张量
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_logical_and(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, and_op);
}

/**
 * @brief 逐元素逻辑或: out = a || b
 * @param a 输入布尔张量
 * @param b 输入布尔张量
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_logical_or(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, or_op);
}

/**
 * @brief 逐元素逻辑异或: out = a ^ b
 * @param a 输入布尔张量
 * @param b 输入布尔张量
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_logical_xor(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general_NUD(a, b, out, xor_op);
}

/**
 * @brief 逐元素逻辑非: out = !a
 * @param a 输入布尔张量
 * @param out 输出布尔张量
 * @return TensorStatus
 */
TensorStatus tensor_logical_not(const Tensor *a, Tensor *out)
{
    return util_unary_op_general_NUD(a, out, not_op);
}