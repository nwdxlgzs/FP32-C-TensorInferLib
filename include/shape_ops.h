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

    /* ==================== 基本形状变换（视图或原地） ==================== */

    /**
     * @brief 原地改变张量形状（要求元素总数不变且数据连续）
     * @param t    目标张量
     * @param ndim 新维度数
     * @param dims 新维度数组
     * @return TensorStatus
     * @note 若张量不连续，需先调用 tensor_contiguous，否则返回错误。
     */
    TensorStatus tensor_reshape(Tensor *t, int ndim, const int *dims);

    /**
     * @brief 创建源张量的视图，并赋予新形状
     * @param dst  输出张量（视图），dst 不应已分配（函数内部会分配）
     * @param src  源张量
     * @param ndim 视图维度数
     * @param dims 视图维度数组
     * @return TensorStatus
     * @note 视图共享数据，源张量的连续性会影响视图的步长计算。
     */
    TensorStatus tensor_reshape_view(Tensor *dst, const Tensor *src, int ndim, const int *dims);

    /**
     * @brief 展平指定轴区间，返回视图
     * @param src        源张量
     * @param start_axis 起始轴（包含）
     * @param end_axis   结束轴（包含）
     * @param dst        输出张量（视图）
     * @return TensorStatus
     * @note 若区间内元素总数溢出或无法合并，返回错误。
     */
    TensorStatus tensor_flatten(const Tensor *src, int start_axis, int end_axis, Tensor *dst);

    /**
     * @brief 移除指定轴上大小为1的维度，返回视图
     * @param src      源张量
     * @param axes     要移除的轴数组，为 NULL 则移除所有大小为1的轴
     * @param num_axes axes 的长度
     * @param dst      输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_squeeze(const Tensor *src, const int *axes, int num_axes, Tensor *dst);

    /**
     * @brief 在指定轴插入大小为1的新维度，返回视图
     * @param src  源张量
     * @param axis 插入位置（支持负索引）
     * @param dst  输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_unsqueeze(const Tensor *src, int axis, Tensor *dst);

    /**
     * @brief 按指定顺序重排轴（视图）
     * @param src  源张量
     * @param axes 新轴顺序数组，长度为 src 的维度数
     * @param dst  输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_transpose_axes(const Tensor *src, const int *axes, Tensor *dst);

    /**
     * @brief 原地交换两个轴（需重新计算步长）
     * @param t     张量
     * @param axis1 轴1
     * @param axis2 轴2
     * @return TensorStatus
     */
    TensorStatus tensor_swapaxes(Tensor *t, int axis1, int axis2);

    /**
     * @brief 沿指定轴反转，返回视图
     * @param src      源张量
     * @param axes     要反转的轴数组，为 NULL 则反转所有轴
     * @param num_axes axes 的长度
     * @param dst      输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_flip(const Tensor *src, const int *axes, int num_axes, Tensor *dst);

    /* ==================== 组合与分割（通常返回新数据） ==================== */

    /**
     * @brief 沿指定轴拼接多个张量
     * @param inputs    输入张量数组
     * @param num_inputs 张量个数
     * @param axis      拼接轴
     * @param output    输出张量（新分配数据）
     * @return TensorStatus
     */
    TensorStatus tensor_concat(const Tensor **inputs, int num_inputs, int axis, Tensor *output);

    /**
     * @brief 沿新轴堆叠多个张量
     * @param inputs    输入张量数组
     * @param num_inputs 张量个数
     * @param axis      新轴位置
     * @param output    输出张量（新分配数据）
     * @return TensorStatus
     */
    TensorStatus tensor_stack(const Tensor **inputs, int num_inputs, int axis, Tensor *output);

    /**
     * @brief 沿指定轴分割张量为多个子张量（视图）
     * @param src        源张量
     * @param axis       分割轴
     * @param sizes      每个子张量在该轴的大小数组
     * @param num_splits 分割份数
     * @param outputs    输出张量数组，需预先分配足够空间（数组元素为 Tensor*），
     *                   函数内部会为每个子张量分配新的 Tensor 结构体，
     *                   调用者必须在使用完毕后对每个 outputs[i] 调用 tensor_destroy 释放。
     * @return TensorStatus
     */
    TensorStatus tensor_split(const Tensor *src, int axis, const int *sizes, int num_splits,
                              Tensor **outputs);

    /**
     * @brief 沿指定轴重复元素，返回新张量
     * @param src     源张量
     * @param axis    重复轴
     * @param repeats 重复次数
     * @param dst     输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_repeat(const Tensor *src, int axis, int repeats, Tensor *dst);

    /**
     * @brief 平铺整个张量，返回新张量
     * @param src  源张量
     * @param reps 每个轴的重复次数，长度需等于 src 的维度
     * @param dst  输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_tile(const Tensor *src, const int *reps, Tensor *dst);

    /* ==================== 填充与累积 ==================== */

    /**
     * @brief 填充模式枚举
     */
    typedef enum
    {
        PAD_CONSTANT,  //!< 常数填充
        PAD_REFLECT,   //!< 反射填充（以边缘为对称轴）
        PAD_REPLICATE, //!< 复制边缘值
        PAD_CIRCULAR   //!< 循环填充
    } PadMode;

    /**
     * @brief 填充，返回新张量
     * @param src           源张量
     * @param pad_widths    每个轴的前后填充宽度数组，长度为 2 * ndim，格式 [before_0, after_0, before_1, after_1, ...]
     * @param mode          填充模式
     * @param constant_value 常数填充的值（仅对 PAD_CONSTANT 有效）
     * @param dst           输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_pad(const Tensor *src, const int *pad_widths, PadMode mode,
                            float constant_value, Tensor *dst);

    /**
     * @brief 沿指定轴累积和，返回新张量
     * @param src  源张量
     * @param axis 轴
     * @param dst  输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_cumsum(const Tensor *src, int axis, Tensor *dst);

    /**
     * @brief 沿指定轴累积积，返回新张量
     * @param src  源张量
     * @param axis 轴
     * @param dst  输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_cumprod(const Tensor *src, int axis, Tensor *dst);


    /**
     * @brief 将张量广播到指定形状（返回视图）
     * @param src  源张量
     * @param ndim 目标维度数
     * @param dims 目标维度数组
     * @param out  输出张量（视图）
     * @return TensorStatus
     * @note 目标形状必须与源张量兼容（广播规则）。
     *       输出张量共享数据，引用计数增加。
     */
    TensorStatus tensor_broadcast_to(const Tensor *src, int ndim, const int *dims, Tensor *out);

    /**
     * @brief 沿指定轴循环滚动张量
     * @param src       源张量
     * @param shifts    每个轴的偏移量数组（正数向右/下，负数向左/上）
     * @param num_axes  轴的数量（若 axes 为 NULL，则必须为 1，表示对所有轴应用相同的 shift）
     * @param axes      轴索引数组，可为 NULL（表示对所有轴滚动）
     * @param out       输出张量（新数据，形状与 src 相同）
     * @return TensorStatus
     * @note 当 axes 为 NULL 时，张量先被展平，然后滚动，再恢复原形状。
     *       当 axes 非 NULL 时，num_axes 必须与 shifts 和 axes 长度一致。
     */
    TensorStatus tensor_roll(const Tensor *src, const int *shifts, int num_axes,
                             const int *axes, Tensor *out);

    /**
     * @brief 移动指定轴到新位置（返回视图）
     * @param src           源张量
     * @param src_axes      要移动的源轴数组
     * @param num_axes      轴的数量
     * @param dst_positions 每个源轴的目标位置数组
     * @param out           输出张量（视图）
     * @return TensorStatus
     * @note 目标位置必须互异且有效，未移动的轴保持原有相对顺序。
     *       输出张量共享数据，引用计数增加。
     */
    TensorStatus tensor_movedim(const Tensor *src, const int *src_axes, int num_axes,
                                const int *dst_positions, Tensor *out);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_SHAPE_OPS_H