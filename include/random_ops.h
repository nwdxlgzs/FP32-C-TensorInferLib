#ifndef TENSOR_RANDOM_OPS_H
#define TENSOR_RANDOM_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file random_ops.h
     * @brief 随机数生成（基于标准C库 rand()）
     */

    /**
     * @brief 设置全局随机种子
     * @param seed 种子值
     * @note 种子会传递给 srand()，后续 rand() 调用将基于此种子生成序列。
     */
    void tensor_random_seed(unsigned int seed);

    /**
     * @brief 用均匀分布填充张量 [low, high)
     * @param t    目标张量
     * @param low  下界（包含）
     * @param high 上界（不包含）
     * @return TensorStatus
     * @note 若 low >= high，返回 TENSOR_ERR_INVALID_PARAM。
     */
    TensorStatus tensor_random_uniform(Tensor *t, float low, float high);

    /**
     * @brief 用正态分布填充张量（Box-Muller 变换）
     * @param t    目标张量
     * @param mean 均值
     * @param std  标准差
     * @return TensorStatus
     */
    TensorStatus tensor_random_normal(Tensor *t, float mean, float std);

    /**
     * @brief 用截断正态分布填充张量（拒绝采样）
     * @param t    目标张量
     * @param mean 均值
     * @param std  标准差
     * @param a    下界
     * @param b    上界
     * @return TensorStatus
     * @note 该函数可能会多次采样，直到所有元素都满足截断条件，效率较低。
     */
    TensorStatus tensor_random_truncated_normal(Tensor *t, float mean, float std, float a, float b);

    /**
     * @brief 用伯努利分布填充张量（生成0或1）
     * @param t 目标张量
     * @param p 概率为1的概率（0 <= p <= 1）
     * @return TensorStatus
     */
    TensorStatus tensor_random_bernoulli(Tensor *t, float p);

    /**
     * @brief 用随机整数填充张量 [low, high)
     * @param t    目标张量
     * @param low  下界（包含）
     * @param high 上界（不包含）
     * @return TensorStatus
     * @note 生成的整数值以 float 类型存储。
     */
    TensorStatus tensor_random_randint(Tensor *t, int low, int high);

    /**
     * @brief 沿第一维打乱张量（Fisher-Yates 算法）
     * @param src 源张量
     * @param dst 输出张量（新数据）
     * @return TensorStatus
     * @note 第一维视为样本维度，对该维度进行随机排列。dst 形状与 src 相同。
     */
    TensorStatus tensor_shuffle(const Tensor *src, Tensor *dst);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_RANDOM_OPS_H