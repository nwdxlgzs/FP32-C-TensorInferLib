#include "tensor.h"
#include "compare_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ==================== 比较操作函数 ==================== */

static float eq_op(float a, float b) { return (a == b) ? 1.0f : 0.0f; }
static float ne_op(float a, float b) { return (a != b) ? 1.0f : 0.0f; }
static float lt_op(float a, float b) { return (a < b) ? 1.0f : 0.0f; }
static float le_op(float a, float b) { return (a <= b) ? 1.0f : 0.0f; }
static float gt_op(float a, float b) { return (a > b) ? 1.0f : 0.0f; }
static float ge_op(float a, float b) { return (a >= b) ? 1.0f : 0.0f; }

/* ==================== 二元比较 API ==================== */

TensorStatus tensor_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, eq_op);
}
TensorStatus tensor_not_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, ne_op);
}
TensorStatus tensor_less(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, lt_op);
}
TensorStatus tensor_less_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, le_op);
}
TensorStatus tensor_greater(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, gt_op);
}
TensorStatus tensor_greater_equal(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, ge_op);
}

/* ==================== 标量比较 API ==================== */

static TensorStatus scalar_compare_op(const Tensor *a, float scalar,
                                      Tensor *out, float (*op)(float, float))
{
    Tensor scalar_tensor;
    scalar_tensor.data = &scalar;
    scalar_tensor.ndim = 0;
    scalar_tensor.dims = NULL;
    scalar_tensor.strides = NULL;
    scalar_tensor.size = 1;
    scalar_tensor.ref_count = NULL;
    scalar_tensor.owns_dims_strides = 0;
    return util_binary_op_general(a, &scalar_tensor, out, op);
}

TensorStatus tensor_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, eq_op);
}
TensorStatus tensor_not_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, ne_op);
}
TensorStatus tensor_less_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, lt_op);
}
TensorStatus tensor_less_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, le_op);
}
TensorStatus tensor_greater_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, gt_op);
}
TensorStatus tensor_greater_equal_scalar(const Tensor *a, float scalar, Tensor *out)
{
    return scalar_compare_op(a, scalar, out, ge_op);
}

/* ==================== 逻辑操作函数 ==================== */

static float and_op(float a, float b) { return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f; }
static float or_op(float a, float b) { return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f; }
static float xor_op(float a, float b) { return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f; }
static float not_op(float a) { return (a == 0.0f) ? 1.0f : 0.0f; }

/* ==================== 逻辑运算 API ==================== */

TensorStatus tensor_logical_and(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, and_op);
}
TensorStatus tensor_logical_or(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, or_op);
}
TensorStatus tensor_logical_xor(const Tensor *a, const Tensor *b, Tensor *out)
{
    return util_binary_op_general(a, b, out, xor_op);
}
TensorStatus tensor_logical_not(const Tensor *a, Tensor *out)
{
    return util_unary_op_general(a, out, not_op);
}