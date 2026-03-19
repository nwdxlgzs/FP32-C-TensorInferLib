#ifndef TENSOR_FFT_OPS_H
#define TENSOR_FFT_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file fft_ops.h
     * @brief 快速傅里叶变换
     */

    /**
     * @brief 快速傅里叶变换（实数输入，返回复数结果）
     * @param src 输入实数张量，最后一维为信号长度
     * @param out 输出复数张量，最后一维为变换后的长度（通常为 n/2+1）
     * @return TensorStatus
     * @note 复数张量用两个连续的 float 表示（实部、虚部交错），形状为 [..., n/2+1, 2]
     * @note 支持非2的幂长度，但是这会回退实现
     */
    TensorStatus tensor_fft_rfft(const Tensor *src, Tensor *out);

    /**
     * @brief 逆傅里叶变换（复数到实数）
     * @param src 输入复数张量，形状 [..., n/2+1, 2]
     * @param n 原始信号长度（若为0，则从输入推断）
     * @param out 输出实数张量，形状 [..., n]
     * @return TensorStatus
     * @note 支持非2的幂长度，但是这会回退实现
     */
    TensorStatus tensor_fft_irfft(const Tensor *src, int n, Tensor *out);

    /**
     * @brief 复数到复数的快速傅里叶变换
     * @param src 输入复数张量，形状 [..., n, 2]
     * @param out 输出复数张量，形状同输入
     * @return TensorStatus
     */
    TensorStatus tensor_fft(const Tensor *src, Tensor *out);

    /**
     * @brief 复数到复数的逆傅里叶变换
     * @param src 输入复数张量，形状 [..., n, 2]
     * @param out 输出复数张量，形状同输入
     * @return TensorStatus
     */
    TensorStatus tensor_ifft(const Tensor *src, Tensor *out);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_FFT_OPS_H