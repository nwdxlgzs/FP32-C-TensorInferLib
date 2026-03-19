#ifndef TENSOR_INDEXING_H
#define TENSOR_INDEXING_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file indexing.h
     * @brief 索引与高级索引操作
     */

    /**
     * @brief 获取单个元素值
     * @param src 源张量
     * @param indices 索引数组，长度等于 src 的维度
     * @param out_value 输出值
     * @return TensorStatus
     * @retval TENSOR_OK 成功
     * @retval TENSOR_ERR_INDEX_OUT_OF_BOUNDS 索引越界
     */
    TensorStatus tensor_get_item(const Tensor *src, const int *indices, float *out_value);

    /**
     * @brief 设置单个元素值（可能触发写时拷贝）
     * @param dst 目标张量
     * @param indices 索引数组
     * @param value 要设置的值
     * @return TensorStatus
     */
    TensorStatus tensor_set_item(Tensor *dst, const int *indices, float value);

    /**
     * @brief 切片获取子张量（返回视图）
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
     * @brief 整数数组索引（高级索引）
     * @param src 源张量
     * @param indices 索引张量数组，每个张量元素为整数（float 类型），形状需兼容
     * @param num_indices 索引张量个数
     * @param out 输出张量（新数据）
     * @return TensorStatus
     */
    TensorStatus tensor_advanced_index(const Tensor *src, const Tensor **indices, int num_indices,
                                       Tensor *out);

    /**
     * @brief 布尔掩码选择
     * @param src 源张量
     * @param mask 布尔掩码张量（0.0/1.0），形状需与 src 相同
     * @param out 输出1维张量（新数据）
     * @return TensorStatus
     */
    TensorStatus tensor_masked_select(const Tensor *src, const Tensor *mask, Tensor *out);

    /**
     * @brief 按索引赋值（可能触发写时拷贝）
     * @param dst 目标张量
     * @param indices 索引张量数组，同 advanced_index
     * @param num_indices 索引个数
     * @param values 要赋值的张量，支持广播
     * @return TensorStatus
     */
    TensorStatus tensor_index_put(Tensor *dst, const Tensor **indices, int num_indices,
                                  const Tensor *values);

    /**
     * @brief 沿指定轴收集元素
     * @param src 源张量
     * @param axis 收集轴
     * @param index 索引张量（元素为整数 float），形状除 axis 外与 src 相同
     * @param out 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_gather(const Tensor *src, int axis, const Tensor *index, Tensor *out);

    /**
     * @brief 沿指定轴将 src 的值放入 dst 的索引位置
     * @param dst 目标张量（可能触发写时拷贝）
     * @param axis 轴
     * @param index 索引张量
     * @param src 源值张量
     * @return TensorStatus
     */
    TensorStatus tensor_scatter(Tensor *dst, int axis, const Tensor *index, const Tensor *src);

    /**
     * @brief 根据线性索引取值（扁平 take）
     * @param src 源张量
     * @param indices 索引张量，元素为整数（float 类型）
     * @param out 输出张量，形状与 indices 相同
     * @return TensorStatus
     */
    TensorStatus tensor_take(const Tensor *src, const Tensor *indices, Tensor *out);

    /**
     * @brief 根据线性索引赋值（扁平 put）
     * @param dst 目标张量（可能触发写时拷贝）
     * @param indices 索引张量，元素为整数（float 类型）
     * @param values 值张量，支持广播到 indices 的形状
     * @param accumulate 非零表示累加，零表示覆盖
     * @return TensorStatus
     */
    TensorStatus tensor_put(Tensor *dst, const Tensor *indices, const Tensor *values, int accumulate);

    /**
     * @brief 返回非零元素的索引
     * @param src 源张量
     * @param out 输出二维张量，形状 (num_nonzero, ndim)
     * @return TensorStatus
     */
    TensorStatus tensor_nonzero(const Tensor *src, Tensor *out);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_INDEXING_H