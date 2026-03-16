#ifndef TENSOR_REDUCE_OPS_H
#define TENSOR_REDUCE_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file reduce_ops.h
     * @brief 归约操作：沿指定轴进行求和、均值、最值等计算
     */

    /**
     * @brief 求和
     * @param x 输入张量
     * @param out 输出张量
     * @param axis 要归约的轴，-1 表示所有轴
     * @param keepdims 非零则保持原维度（大小为1）
     * @return TensorStatus
     */
    TensorStatus tensor_sum(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 求平均值
     */
    TensorStatus tensor_mean(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 求乘积
     */
    TensorStatus tensor_prod(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 求最大值
     */
    TensorStatus tensor_max(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 求最小值
     */
    TensorStatus tensor_min(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 求最大值索引（返回整数索引作为浮点数）
     */
    TensorStatus tensor_argmax(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 求最小值索引
     */
    TensorStatus tensor_argmin(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 逻辑或（非零视为 True）
     */
    TensorStatus tensor_any(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 逻辑与
     */
    TensorStatus tensor_all(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 方差
     * @param unbiased 非零则使用样本方差（除以 n-1），否则除以 n
     */
    TensorStatus tensor_var(const Tensor *x, Tensor *out, int axis, int keepdims, int unbiased);

    /**
     * @brief 标准差
     * @param unbiased 同方差
     */
    TensorStatus tensor_std(const Tensor *x, Tensor *out, int axis, int keepdims, int unbiased);

    /**
     * @brief p 范数
     * @param p 范数阶数，p=0 表示非零元素个数
     */
    TensorStatus tensor_norm(const Tensor *x, Tensor *out, int axis, int keepdims, float p);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_REDUCE_OPS_H