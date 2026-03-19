#ifndef TENSOR_SEARCH_OPS_H
#define TENSOR_SEARCH_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file search_ops.h
     * @brief 搜索与排序操作
     */

    /**
     * @brief 沿指定轴对张量进行排序（升序）
     * @param src  输入张量
     * @param axis 排序轴，-1 表示展平后排序（返回一维张量）
     * @param out  输出排序后的张量，形状与 src 相同（若 axis=-1，则为一维）
     * @return TensorStatus
     * @note 排序算法为快速排序（不稳定），对非连续张量会先转为连续再处理。
     */
    TensorStatus tensor_sort(const Tensor *src, int axis, Tensor *out);

    /**
     * @brief 返回沿指定轴排序后的索引（升序）
     * @param src  输入张量
     * @param axis 排序轴
     * @param out  输出索引张量，元素类型为 float（存储整数值），形状同 src
     * @return TensorStatus
     */
    TensorStatus tensor_argsort(const Tensor *src, int axis, Tensor *out);

    /**
     * @brief 返回张量中的唯一元素（升序）
     * @param src 输入张量
     * @param out 输出一维张量，包含所有唯一值（按升序排列）
     * @return TensorStatus
     */
    TensorStatus tensor_unique(const Tensor *src, Tensor *out);

    /**
     * @brief 在有序张量中查找元素应插入的位置（保持顺序）
     * @param sorted 有序张量（升序）
     * @param values 待查找的值（张量，支持广播）
     * @param right  若为非零，返回最右侧的插入位置（即二分查找的 upper_bound）；
     *               若为零，返回最左侧的插入位置（lower_bound）。
     * @param out    输出索引张量，形状与 values 广播后相同
     * @return TensorStatus
     */
    TensorStatus tensor_searchsorted(const Tensor *sorted, const Tensor *values, int right, Tensor *out);

    /**
     * @brief 沿指定轴返回最大的 k 个值和它们的索引
     * @param src 输入张量
     * @param k 要返回的元素个数
     * @param axis 轴，-1 表示展平后计算（返回一维张量）
     * @param largest 非零表示返回最大的 k 个，零表示返回最小的 k 个
     * @param sorted 非零表示返回的值按降序（如果 largest=1）或升序（如果 largest=0）排序；零表示不保证顺序（当前实现始终排序）
     * @param values 输出值张量，形状：除 axis 外与 src 相同，但 axis 维度大小变为 k
     * @param indices 输出索引张量，形状同 values，元素为 float 类型索引
     * @return TensorStatus
     */
    TensorStatus tensor_topk(const Tensor *src, int k, int axis, int largest, int sorted,
                             Tensor *values, Tensor *indices);

    /**
     * @brief 沿指定轴返回第 k 小的值和索引（k 从 1 开始）
     * @param src 输入张量
     * @param k 第几个最小值（1-based）
     * @param axis 轴，-1 表示展平后计算（返回标量或一维张量）
     * @param keepdims 非零则保持原维度（大小为1）
     * @param values 输出值张量
     * @param indices 输出索引张量，形状同 values
     * @return TensorStatus
     */
    TensorStatus tensor_kthvalue(const Tensor *src, int k, int axis, int keepdims,
                                 Tensor *values, Tensor *indices);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_SEARCH_OPS_H