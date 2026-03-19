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

    /* ==================== 矩阵乘法相关 ==================== */

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
     * @note 矩阵 A、B、C 的内存区域不应重叠。
     * @note 分块大小 TILE 定义为 TENSOR_MATMUL_TILE (64)，可根据目标 CPU 缓存调整。
     * @warning 函数内部会清零矩阵 C，因此调用前无需对 C 初始化；但若部分结果需累积，
     *          请勿使用此函数（它不执行累加，而是直接覆盖）。
     */
    void matmul_tiled(int M, int N, int K,
                      const float *a, const float *b,
                      float *c);

    /* ==================== 转置/形状操作 ==================== */

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

    /* ==================== 矩阵分解与求解 ==================== */

    /**
     * @brief 矩阵求逆（仅支持方阵，朴素高斯消元）
     * @param src 输入方阵（2维）
     * @param out 输出方阵
     * @return TensorStatus
     */
    TensorStatus tensor_inv(const Tensor *src, Tensor *out);

    /**
     * @brief 计算方阵的行列式
     * @param src 输入方阵（2维）
     * @param out 输出标量张量（0维或1维长度为1）
     * @return TensorStatus
     * @note 奇异矩阵会返回 TENSOR_ERR_SINGULAR_MATRIX
     */
    TensorStatus tensor_det(const Tensor *src, Tensor *out);

    /**
     * @brief 计算实对称矩阵的特征值和特征向量（使用雅可比方法）
     * @param src 输入实对称方阵（2维）
     * @param eigvals 输出特征值张量（1维，长度为 n）
     * @param eigvecs 输出特征向量张量（2维，n×n，列为特征向量）
     * @return TensorStatus
     */
    TensorStatus tensor_eigh(const Tensor *src, Tensor *eigvals, Tensor *eigvecs);

    /**
     * @brief 奇异值分解：A = U * S * V^T
     * @param src 输入矩阵（2维）
     * @param U 输出左奇异矩阵，形状 (M, M) 或 (M, K) 取决于 full
     * @param S 输出奇异值向量，形状 (K,)
     * @param V 输出右奇异矩阵，形状 (N, N) 或 (N, K)
     * @param full 非零表示计算完整矩阵，否则计算经济分解
     * @return TensorStatus
     */
    TensorStatus tensor_svd(const Tensor *src, Tensor *U, Tensor *S, Tensor *V, int full);

    /**
     * @brief QR 分解：A = Q * R
     * @param src 输入矩阵（2维）
     * @param Q 输出正交矩阵（或经济形式的 Q）
     * @param R 输出上三角矩阵
     * @param reduced 非零表示经济分解（Q 形状 (M, min(M,N))，R 形状 (min(M,N), N)）
     * @return TensorStatus
     */
    TensorStatus tensor_qr(const Tensor *src, Tensor *Q, Tensor *R, int reduced);

    /**
     * @brief Cholesky 分解：A = L * L^T（A 对称正定）
     * @param src 输入对称正定方阵
     * @param out 输出下三角矩阵 L
     * @return TensorStatus
     */
    TensorStatus tensor_cholesky(const Tensor *src, Tensor *out);

    /**
     * @brief 解线性方程组 AX = B
     * @param A 系数矩阵（2维方阵）
     * @param B 右侧张量（可与 A 共享批量维度）
     * @param X 输出解张量，形状与 B 相同（但最后一维为 A 的列数）
     * @return TensorStatus
     */
    TensorStatus tensor_solve(const Tensor *A, const Tensor *B, Tensor *X);

    /**
     * @brief 一般实矩阵的特征值分解（返回复特征值）
     * @param src         输入方阵（2维，n×n）
     * @param eigvals_real 输出特征值实部（1维，长度 n）
     * @param eigvals_imag 输出特征值虚部（1维，长度 n）
     * @param eigvecs_real 输出特征向量实部（2维，n×n，列为特征向量）
     * @param eigvecs_imag 输出特征向量虚部（2维，n×n）
     * @return TensorStatus
     * @note 若特征值为实数，对应虚部为0；若为复数，则共轭对按实部升序排列。
     */
    TensorStatus tensor_eig(const Tensor *src,
                            Tensor *eigvals_real, Tensor *eigvals_imag,
                            Tensor *eigvecs_real, Tensor *eigvecs_imag);

    /**
     * @brief 线性最小二乘求解：min ||A X - B||₂
     * @param A 系数矩阵（2维，m×n）
     * @param B 右侧张量（2维，m×k，或1维 m）
     * @param X 输出解（2维，n×k，或1维 n）
     * @return TensorStatus
     * @note 使用 SVD 分解，对任意形状（超定/欠定/满秩/亏秩）均有效。
     */
    TensorStatus tensor_lstsq(const Tensor *A, const Tensor *B, Tensor *X);

    /**
     * @brief 计算矩阵的秩（基于 SVD 的数值秩）
     * @param src 输入矩阵（2维）
     * @param out 输出标量张量（0维或1维长度为1）
     * @param tol 容差（若 <=0，则自动取 max(m,n) * eps * max(σ)）
     * @return TensorStatus
     */
    TensorStatus tensor_matrix_rank(const Tensor *src, float tol, Tensor *out);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_LINALG_OPS_H