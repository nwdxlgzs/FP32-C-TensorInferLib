#ifndef TENSOR_REDUCE_OPS_H
#define TENSOR_REDUCE_OPS_H

#include "tensor.h"

/**
 * @file reduce_ops.h
 * @brief 归约操作：沿指定轴进行求和、均值、最值等计算
 */

#ifdef __cplusplus
extern "C"
{
#endif

    /* ==================== 基础归约 ==================== */

    /**
     * @brief 求和
     * @param x 输入张量
     * @param out 输出张量
     * @param axis 要归约的轴，-1 表示所有轴
     * @param keepdims 非零则保持原维度（大小为1）
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

    /* ==================== 方差与标准差 ==================== */

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

    /* ==================== 分位数与排序统计 ==================== */

    /**
     * @brief 分位数插值方法
     */
    typedef enum
    {
        QUANTILE_LINEAR,   // 线性插值 (默认)
        QUANTILE_LOWER,    // 较低值
        QUANTILE_HIGHER,   // 较高值
        QUANTILE_MIDPOINT, // 中点
        QUANTILE_NEAREST   // 最近邻
    } QuantileInterp;

    /**
     * @brief 沿指定轴计算中位数
     * @param x 输入张量
     * @param out 输出张量
     * @param axis 要归约的轴，-1 表示所有轴
     * @param keepdims 非零则保持原维度（大小为1）
     */
    TensorStatus tensor_median(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 沿指定轴计算众数（出现次数最多的值）
     * @param x 输入张量
     * @param out 输出众数值张量
     * @param axis 要归约的轴，-1 表示所有轴
     * @param keepdims 非零则保持原维度（大小为1）
     * @note 若有多个众数，返回最小的那个
     */
    TensorStatus tensor_mode(const Tensor *x, Tensor *out, int axis, int keepdims);

    /**
     * @brief 沿指定轴计算分位数（q 为标量）
     * @param x 输入张量
     * @param q 分位数标量张量（0维或1维长度为1），取值范围 [0,1]
     * @param axis 要归约的轴，-1 表示所有轴
     * @param keepdims 非零则保持原维度（大小为1）
     * @param interp 插值方法（见 QuantileInterp）
     * @param out 输出张量
     */
    TensorStatus tensor_quantile(const Tensor *x, const Tensor *q, int axis, int keepdims,
                                 QuantileInterp interp, Tensor *out);

    /* ==================== 累积操作 ==================== */

    /**
     * @brief 沿指定轴计算累积最大值
     * @param src  输入张量
     * @param axis 轴（必须有效）
     * @param dst  输出张量，形状与 src 相同
     * @return TensorStatus
     */
    TensorStatus tensor_cummax(const Tensor *src, int axis, Tensor *dst);

    /**
     * @brief 沿指定轴计算累积最小值
     */
    TensorStatus tensor_cummin(const Tensor *src, int axis, Tensor *dst);

    /**
     * @brief 沿指定轴计算累积 log(exp(x1)+exp(x2)+...) （数值稳定版）
     */
    TensorStatus tensor_logcumsumexp(const Tensor *src, int axis, Tensor *dst);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_REDUCE_OPS_H