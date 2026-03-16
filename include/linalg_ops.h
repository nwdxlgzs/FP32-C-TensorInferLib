#ifndef TENSOR_LINALG_OPS_H
#define TENSOR_LINALG_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file linalg_ops.h
     * @brief 线性代数运算
     */
#ifndef TENSOR_MATMUL_TILE
#define TENSOR_MATMUL_TILE 64
#endif
    /**
     * @brief 矩阵乘法（支持批量广播）
     *
     * a 形状 [..., M, K], b 形状 [..., K, N] -> out 形状 [..., M, N]
     *
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 批量矩阵乘法（严格 3 维）
     *
     * a 形状 [batch, M, K], b 形状 [batch, K, N] -> out 形状 [batch, M, N]
     *
     * @param a 输入张量
     * @param b 输入张量
     * @param out 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_bmm(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 向量点积（两个 1 维张量）
     * @param a 1维张量
     * @param b 1维张量
     * @param out 输出标量张量（0维或1维长度为1）
     * @return TensorStatus
     */
    TensorStatus tensor_dot(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 向量外积（两个 1 维张量）
     * @param a 1维张量 (m)
     * @param b 1维张量 (n)
     * @param out 输出2维张量 (m, n)
     * @return TensorStatus
     */
    TensorStatus tensor_outer(const Tensor *a, const Tensor *b, Tensor *out);

    /**
     * @brief 张量缩并（广义点积）
     * @param a 输入张量
     * @param b 输入张量
     * @param axes_a 要缩并的 a 的轴索引数组
     * @param axes_b 要缩并的 b 的轴索引数组
     * @param naxes 轴对数
     * @param out 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_tensordot(const Tensor *a, const Tensor *b,
                                  const int *axes_a, const int *axes_b, int naxes,
                                  Tensor *out);

    /**
     * @brief 矩阵转置（交换最后两维）
     * @param src 输入张量（至少2维）
     * @param out 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_transpose(const Tensor *src, Tensor *out);

    /**
     * @brief 一般转置（按指定顺序重排轴）
     * @param src 输入张量
     * @param axes 新轴的顺序，长度为 src 的维度数
     * @param out 输出张量（视图）
     * @return TensorStatus
     */
    TensorStatus tensor_permute(const Tensor *src, const int *axes, Tensor *out);

    /**
     * @brief 提取对角线（2维输入）或创建对角阵（1维输入）
     * @param src 输入张量
     * @param out 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_diag(const Tensor *src, Tensor *out);

    /**
     * @brief 迹（沿 axis1 和 axis2 的对角线和）
     * @param src 输入张量
     * @param axis1 第一个轴
     * @param axis2 第二个轴
     * @param out 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_trace(const Tensor *src, int axis1, int axis2, Tensor *out);

    /**
     * @brief 矩阵求逆（仅支持方阵，朴素高斯消元）
     * @param src 输入方阵（2维）
     * @param out 输出方阵
     * @return TensorStatus
     */
    TensorStatus tensor_inv(const Tensor *src, Tensor *out);

    /**
     * @brief 分块矩阵乘法（行主序）
     *
     * 计算 C = A * B，其中 A 是 M×K 矩阵，B 是 K×N 矩阵，C 是 M×N 矩阵。
     * 采用分块（tiled）算法，通过将矩阵划分为适合 CPU 缓存的小块，提高数据局部性，
     * 从而获得更好的性能。该实现完全使用标准 C，不依赖任何硬件特性或外部库，
     * 具有平台无关性。
     *
     * @param M 矩阵 A 的行数，也是矩阵 C 的行数（必须 > 0）
     * @param N 矩阵 B 的列数，也是矩阵 C 的列数（必须 > 0）
     * @param K 矩阵 A 的列数，也是矩阵 B 的行数（必须 > 0）
     * @param a 指向矩阵 A 数据的指针，长度为 M * K，按行主序存储
     * @param b 指向矩阵 B 数据的指针，长度为 K * N，按行主序存储
     * @param c 指向矩阵 C 数据的指针，长度为 M * N，按行主序存储；函数执行前无需初始化，内部会将结果写入
     *
     * @note 矩阵 A、B、C 的内存区域不应重叠，建议使用 `restrict` 指针（已隐含假设）。
     * @note 为获得最佳性能，建议使用对齐的内存（如通过 `aligned_alloc` 分配），
     *       并确保矩阵数据连续（无间隔）。
     * @note 分块大小 TILE 在定义为 TENSOR_MATMUL_TILE(64)，可根据目标 CPU 的缓存大小调整。
     * @warning 函数内部会清零矩阵 C，因此调用前无需对 C 初始化；但若部分结果需累积，
     *          请勿使用此函数（它不执行累加，而是直接覆盖）。
     */
    void matmul_tiled(int M, int N, int K,
                      const float *a, const float *b,
                      float *c);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_LINALG_OPS_H