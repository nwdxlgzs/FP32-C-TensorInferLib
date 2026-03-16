#ifndef TENSOR_SHAPE_OPS_H
#define TENSOR_SHAPE_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file shape_ops.h
     * @brief 形状操作：变形、拼接、切片、重复等
     */

    /**
     * @brief 原地改变形状（要求元素总数不变且数据连续）
     * @param t 张量
     * @param ndim 新维度数
     * @param dims 新维度数组
     * @return TensorStatus
     */
    TensorStatus tensor_reshape(Tensor *t, int ndim, const int *dims);

    /**
     * @brief 创建视图（共享数据）并存储到 dst
     * @param src 源张量
     * @param ndim 视图维度数
     * @param dims 视图维度数组
     * @param dst 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_reshape_view(const Tensor *src, int ndim, const int *dims, Tensor *dst);

    /**
     * @brief 展平指定轴区间，返回视图
     * @param src 源张量
     * @param start_axis 起始轴
     * @param end_axis 结束轴（包含）
     * @param dst 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_flatten(const Tensor *src, int start_axis, int end_axis, Tensor *dst);

    /**
     * @brief 移除指定轴（大小为1），返回视图
     * @param src 源张量
     * @param axes 要移除的轴数组，为 NULL 则移除所有大小为1的轴
     * @param num_axes axes 的长度
     * @param dst 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_squeeze(const Tensor *src, const int *axes, int num_axes, Tensor *dst);

    /**
     * @brief 在指定轴插入大小为1的新维度，返回视图
     * @param src 源张量
     * @param axis 插入位置（支持负索引）
     * @param dst 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_unsqueeze(const Tensor *src, int axis, Tensor *dst);

    /**
     * @brief 沿指定轴拼接多个张量
     * @param inputs 输入张量数组
     * @param num_inputs 张量个数
     * @param axis 拼接轴
     * @param output 输出张量（新分配数据）
     * @return TensorStatus
     */
    TensorStatus tensor_concat(const Tensor **inputs, int num_inputs, int axis, Tensor *output);

    /**
     * @brief 沿新轴堆叠多个张量
     * @param inputs 输入张量数组
     * @param num_inputs 张量个数
     * @param axis 新轴位置
     * @param output 输出张量（新分配数据）
     * @return TensorStatus
     */
    TensorStatus tensor_stack(const Tensor **inputs, int num_inputs, int axis, Tensor *output);

    /**
     * @brief 沿指定轴分割张量为多个子张量（视图）
     * @param src 源张量
     * @param axis 分割轴
     * @param sizes 每个子张量在该轴的大小数组
     * @param num_splits 分割份数
     * @param outputs 输出张量数组（视图），需预先分配足够空间
     * @return TensorStatus
     */
    TensorStatus tensor_split(const Tensor *src, int axis, const int *sizes, int num_splits,
                              Tensor **outputs);

    /**
     * @brief 切片（支持负索引、步长），返回视图
     * @param src 源张量
     * @param starts 每个轴的起始索引
     * @param ends 每个轴的结束索引（不包含）
     * @param steps 每个轴的步长，为 NULL 则步长为1
     * @param dst 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_slice(const Tensor *src, const int *starts, const int *ends,
                              const int *steps, Tensor *dst);

    /**
     * @brief 沿指定轴重复元素，返回新张量
     * @param src 源张量
     * @param axis 重复轴
     * @param repeats 重复次数
     * @param dst 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_repeat(const Tensor *src, int axis, int repeats, Tensor *dst);

    /**
     * @brief 平铺整个张量，返回新张量
     * @param src 源张量
     * @param reps 每个轴的重复次数，长度需等于 src 的维度
     * @param dst 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_tile(const Tensor *src, const int *reps, Tensor *dst);

    /**
     * @brief 按指定顺序重排轴（视图）
     * @param src 源张量
     * @param axes 新轴顺序数组，长度为 src 的维度
     * @param dst 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_transpose_axes(const Tensor *src, const int *axes, Tensor *dst);

    /**
     * @brief 原地交换两个轴（需重新计算步长）
     * @param t 张量
     * @param axis1 轴1
     * @param axis2 轴2
     * @return TensorStatus
     */
    TensorStatus tensor_swapaxes(Tensor *t, int axis1, int axis2);

    /**
     * @brief 沿指定轴反转，返回视图
     * @param src 源张量
     * @param axes 要反转的轴数组，为 NULL 则反转所有轴
     * @param num_axes axes 的长度
     * @param dst 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_flip(const Tensor *src, const int *axes, int num_axes, Tensor *dst);

    typedef enum
    {
        PAD_CONSTANT,  // 常数填充
        PAD_REFLECT,   // 反射填充
        PAD_REPLICATE, // 复制边缘
        PAD_CIRCULAR   // 循环填充
    } PadMode;

    /**
     * @brief 填充，返回新张量
     * @param src 源张量
     * @param pad_widths 每个轴的前后填充宽度数组，长度为 2 * ndim，格式 [before_0, after_0, before_1, after_1, ...]
     * @param mode 填充模式
     * @param constant_value 常数填充的值
     * @param dst 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_pad(const Tensor *src, const int *pad_widths, PadMode mode,
                            float constant_value, Tensor *dst);

    /**
     * @brief 沿指定轴累积和，返回新张量
     * @param src 源张量
     * @param axis 轴
     * @param dst 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_cumsum(const Tensor *src, int axis, Tensor *dst);

    /**
     * @brief 沿指定轴累积积，返回新张量
     * @param src 源张量
     * @param axis 轴
     * @param dst 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_cumprod(const Tensor *src, int axis, Tensor *dst);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_SHAPE_OPS_H