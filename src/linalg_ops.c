#include "tensor.h"
#include "linalg_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/**
 * @file linalg_ops.c
 * @brief 线性代数运算的实现
 *
 * 包含矩阵乘法（支持广播）、向量点积/外积、张量缩并（tensordot）、
 * 转置/形状操作、矩阵分解（LU、Cholesky、QR、SVD、Eigh）、
 * 求解线性方程组等。
 */

/* ==================== 内部辅助函数 ==================== */

/**
 * @brief 检查张量的矩阵部分（从指定轴开始）是否连续
 * @param t          张量
 * @param start_axis 起始轴（通常为批量维度之后）
 * @return 1 表示连续，0 表示不连续
 */
static int is_contiguous_matrix_part(const Tensor *t, int start_axis)
{
    if (!t)
        return 0;
    int ndim = t->ndim;
    if (t->strides == NULL)
        return 1; // 显式连续

    int expected = 1;
    for (int i = ndim - 1; i >= start_axis; --i)
    {
        if (t->dims[i] == 1)
            continue; // 广播维度不影响连续性
        if (t->strides[i] != expected)
            return 0;
        expected *= t->dims[i];
    }
    return 1;
}

/**
 * @brief 正交化向量 v，使其与 Q 的前 n_orth 列正交，并归一化
 * @param Q       Q 矩阵，列主序存储（第 j 列起始地址 = Q + j * m）
 * @param m       行数
 * @param n_orth  已正交化的列数
 * @param v       输入/输出向量（长度 m）
 * @param tol     容差，用于判断向量是否足够大
 * @return 1 成功，0 失败
 */
static int find_orthogonal_vector(const double *Q, int m, int n_orth, double *v, double tol)
{
    // 尝试随机向量（最多 10 次）
    for (int attempt = 0; attempt < 10; ++attempt)
    {
        util_random_double_vector(v, m); // 生成 [-1,1] 随机向量
        // Gram-Schmidt 正交化
        for (int j = 0; j < n_orth; ++j)
        {
            double dot = 0.0;
            for (int i = 0; i < m; ++i)
            {
                dot += Q[j * m + i] * v[i];
            }
            for (int i = 0; i < m; ++i)
            {
                v[i] -= dot * Q[j * m + i];
            }
        }
        double norm = 0.0;
        for (int i = 0; i < m; ++i)
            norm += v[i] * v[i];
        if (norm > tol * tol)
        {
            norm = sqrt(norm);
            for (int i = 0; i < m; ++i)
                v[i] /= norm;
            return 1;
        }
    }

    // 尝试标准基
    for (int k = 0; k < m; ++k)
    {
        for (int i = 0; i < m; ++i)
            v[i] = (i == k) ? 1.0 : 0.0;
        for (int j = 0; j < n_orth; ++j)
        {
            double dot = 0.0;
            for (int i = 0; i < m; ++i)
            {
                dot += Q[j * m + i] * v[i];
            }
            for (int i = 0; i < m; ++i)
            {
                v[i] -= dot * Q[j * m + i];
            }
        }
        double norm = 0.0;
        for (int i = 0; i < m; ++i)
            norm += v[i] * v[i];
        if (norm > tol * tol)
        {
            norm = sqrt(norm);
            for (int i = 0; i < m; ++i)
                v[i] /= norm;
            return 1;
        }
    }
    return 0; // 失败（理论上不会发生）
}

/* ==================== 矩阵乘法 (支持批量广播) ==================== */

/**
 * @brief 分块矩阵乘法（核心实现）
 * @param M A 的行数，C 的行数
 * @param N B 的列数，C 的列数
 * @param K A 的列数，B 的行数
 * @param a 矩阵 A 数据指针（行主序，长度 M*K）
 * @param b 矩阵 B 数据指针（行主序，长度 K*N）
 * @param c 矩阵 C 数据指针（行主序，长度 M*N），函数执行前未初始化，内部将结果写入（累加模式）
 * @note 此函数假设 c 已清零（或需要累加），调用前应确保 c 缓冲区清零。
 */
void matmul_tiled(int M, int N, int K,
                  const float *a, const float *b,
                  float *c)
{
    const int tile = TENSOR_MATMUL_TILE;
    for (int i0 = 0; i0 < M; i0 += tile)
    {
        int imax = (i0 + tile < M) ? i0 + tile : M;
        for (int j0 = 0; j0 < N; j0 += tile)
        {
            int jmax = (j0 + tile < N) ? j0 + tile : N;
            for (int k0 = 0; k0 < K; k0 += tile)
            {
                int kmax = (k0 + tile < K) ? k0 + tile : K;
                for (int i = i0; i < imax; i++)
                {
                    for (int k = k0; k < kmax; k++)
                    {
                        float aik = a[i * K + k];
                        for (int j = j0; j < jmax; j++)
                        {
                            c[i * N + j] += aik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief 矩阵乘法（支持批量广播）
 *
 * 根据输入维度自动处理：
 * - 两个 1 维张量：向量点积
 * - 1 维 × 2+ 维：向量与矩阵乘法
 * - 2+ 维 × 1 维：矩阵与向量乘法
 * - 2+ 维 × 2+ 维：批量矩阵乘法，支持广播
 *
 * @param a   输入张量 A
 * @param b   输入张量 B
 * @param out 输出张量
 * @return TensorStatus
 */
TensorStatus tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out)
{
    if (!a || !b || !out)
        return TENSOR_ERR_NULL_PTR;

    int a_ndim = a->ndim;
    int b_ndim = b->ndim;
    if (a_ndim < 1 || b_ndim < 1)
        return TENSOR_ERR_INVALID_PARAM;

    /* ---------- 处理一维情况 ---------- */
    if (a_ndim == 1 && b_ndim == 1)
    {
        // 向量点积
        if (a->size != b->size)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (a->size == 0)
            return TENSOR_ERR_INVALID_PARAM;
        if (out->ndim != 0 && !(out->ndim == 1 && out->dims[0] == 1))
            return TENSOR_ERR_SHAPE_MISMATCH;

        TensorStatus status = tensor_make_unique(out);
        if (status != TENSOR_OK)
            return status;

        double sum = 0.0;
        for (size_t i = 0; i < a->size; i++)
            sum += (double)a->data[i] * b->data[i];
        out->data[0] = (float)sum;
        return TENSOR_OK;
    }

    if (a_ndim == 1 && b_ndim >= 2)
    {
        int K = (int)a->size;
        int b_K = b->dims[b_ndim - 2];
        if (K != b_K)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (K <= 0 || b->dims[b_ndim - 1] <= 0)
            return TENSOR_ERR_INVALID_PARAM;

        int out_ndim = b_ndim - 1;
        int out_dims[TENSOR_MAX_DIM];
        for (int i = 0; i < out_ndim; i++)
            out_dims[i] = b->dims[i];
        out_dims[out_ndim - 1] = b->dims[b_ndim - 1];

        if (out->ndim != out_ndim)
            return TENSOR_ERR_SHAPE_MISMATCH;
        for (int i = 0; i < out_ndim; i++)
            if (out->dims[i] != out_dims[i])
                return TENSOR_ERR_SHAPE_MISMATCH;

        TensorStatus status = tensor_make_unique(out);
        if (status != TENSOR_OK)
            return status;

        int a_strides[1], b_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
        util_get_effective_strides(a, a_strides);
        util_get_effective_strides(b, b_strides);
        util_get_effective_strides(out, out_strides);

        int batch_ndim = out_ndim - 1;
        int batch_dims[TENSOR_MAX_DIM];
        for (int i = 0; i < batch_ndim; i++)
            batch_dims[i] = out_dims[i];
        int batch_coords[TENSOR_MAX_DIM] = {0};
        int b_batch_strides[TENSOR_MAX_DIM];
        for (int i = 0; i < batch_ndim; i++)
            b_batch_strides[i] = b_strides[i];
        int out_batch_strides[TENSOR_MAX_DIM];
        for (int i = 0; i < batch_ndim; i++)
            out_batch_strides[i] = out_strides[i];

        int b_stride_k = b_strides[batch_ndim];
        int b_stride_n = b_strides[batch_ndim + 1];
        int out_stride_n = out_strides[batch_ndim];

        while (1)
        {
            size_t b_base = 0, out_base = 0;
            for (int d = 0; d < batch_ndim; d++)
            {
                b_base += batch_coords[d] * b_batch_strides[d];
                out_base += batch_coords[d] * out_batch_strides[d];
            }

            int N = out_dims[out_ndim - 1];
            for (int n = 0; n < N; n++)
            {
                double sum = 0.0;
                for (int k = 0; k < K; k++)
                {
                    size_t a_off = k * a_strides[0];
                    size_t b_off = b_base + k * b_stride_k + n * b_stride_n;
                    sum += (double)a->data[a_off] * b->data[b_off];
                }
                out->data[out_base + n * out_stride_n] = (float)sum;
            }

            if (util_increment_coords(batch_coords, batch_dims, batch_ndim))
                break;
        }
        return TENSOR_OK;
    }

    if (a_ndim >= 2 && b_ndim == 1)
    {
        int K = (int)b->size;
        int a_K = a->dims[a_ndim - 1];
        if (K != a_K)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (K <= 0 || a->dims[a_ndim - 2] <= 0)
            return TENSOR_ERR_INVALID_PARAM;

        int out_ndim = a_ndim - 1;
        int out_dims[TENSOR_MAX_DIM];
        for (int i = 0; i < out_ndim; i++)
            out_dims[i] = a->dims[i];

        if (out->ndim != out_ndim)
            return TENSOR_ERR_SHAPE_MISMATCH;
        for (int i = 0; i < out_ndim; i++)
            if (out->dims[i] != out_dims[i])
                return TENSOR_ERR_SHAPE_MISMATCH;

        TensorStatus status = tensor_make_unique(out);
        if (status != TENSOR_OK)
            return status;

        int a_strides[TENSOR_MAX_DIM], b_strides[1], out_strides[TENSOR_MAX_DIM];
        util_get_effective_strides(a, a_strides);
        util_get_effective_strides(b, b_strides);
        util_get_effective_strides(out, out_strides);

        int batch_ndim = out_ndim - 1;
        int batch_dims[TENSOR_MAX_DIM];
        for (int i = 0; i < batch_ndim; i++)
            batch_dims[i] = out_dims[i];
        int a_batch_strides[TENSOR_MAX_DIM];
        for (int i = 0; i < batch_ndim; i++)
            a_batch_strides[i] = a_strides[i];
        int out_batch_strides[TENSOR_MAX_DIM];
        for (int i = 0; i < batch_ndim; i++)
            out_batch_strides[i] = out_strides[i];

        int a_stride_m = a_strides[batch_ndim];
        int a_stride_k = a_strides[batch_ndim + 1];
        int out_stride_m = out_strides[batch_ndim];

        int batch_coords[TENSOR_MAX_DIM] = {0};
        while (1)
        {
            size_t a_base = 0, out_base = 0;
            for (int d = 0; d < batch_ndim; d++)
            {
                a_base += batch_coords[d] * a_batch_strides[d];
                out_base += batch_coords[d] * out_batch_strides[d];
            }

            int M = out_dims[out_ndim - 1];
            for (int m = 0; m < M; m++)
            {
                double sum = 0.0;
                for (int k = 0; k < K; k++)
                {
                    size_t a_off = a_base + m * a_stride_m + k * a_stride_k;
                    size_t b_off = k * b_strides[0];
                    sum += (double)a->data[a_off] * b->data[b_off];
                }
                out->data[out_base + m * out_stride_m] = (float)sum;
            }

            if (util_increment_coords(batch_coords, batch_dims, batch_ndim))
                break;
        }
        return TENSOR_OK;
    }

    /* ---------- 批量矩阵乘法 (a_ndim >=2, b_ndim >=2) ---------- */
    int a_batch_ndim = a_ndim - 2;
    int b_batch_ndim = b_ndim - 2;

    int a_M = a->dims[a_ndim - 2];
    int a_K = a->dims[a_ndim - 1];
    int b_K = b->dims[b_ndim - 2];
    int b_N = b->dims[b_ndim - 1];

    if (a_K != b_K)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (a_M <= 0 || a_K <= 0 || b_N <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    /* 批量维度广播 */
    int batch_ndim;
    int batch_dims[TENSOR_MAX_DIM];
    if (!util_broadcast_shape(a->dims, a_batch_ndim, b->dims, b_batch_ndim,
                              batch_dims, &batch_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    for (int i = 0; i < batch_ndim; i++)
        if (batch_dims[i] <= 0)
            return TENSOR_ERR_INVALID_PARAM;

    /* 输出形状 = 广播批量 + [M, N] */
    int out_ndim = batch_ndim + 2;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; i++)
        out_dims[i] = batch_dims[i];
    out_dims[batch_ndim] = a_M;
    out_dims[batch_ndim + 1] = b_N;

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; i++)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    /* 获取有效步长 */
    int a_eff_strides[TENSOR_MAX_DIM], b_eff_strides[TENSOR_MAX_DIM];
    int out_eff_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(a, a_eff_strides);
    util_get_effective_strides(b, b_eff_strides);
    util_get_effective_strides(out, out_eff_strides);

    /* 填充步长到输出维度 */
    int a_padded[TENSOR_MAX_DIM], b_padded[TENSOR_MAX_DIM];
    util_fill_padded_strides(a, out_ndim, out_dims, a_padded);
    util_fill_padded_strides(b, out_ndim, out_dims, b_padded);

    /* 提取批量部分步长和矩阵步长 */
    int a_batch_strides[TENSOR_MAX_DIM], b_batch_strides[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; i++)
    {
        a_batch_strides[i] = a_padded[i];
        b_batch_strides[i] = b_padded[i];
    }
    int a_stride_i = a_padded[batch_ndim];
    int a_stride_k = a_padded[batch_ndim + 1];
    int b_stride_k = b_padded[batch_ndim];
    int b_stride_j = b_padded[batch_ndim + 1];

    int out_batch_strides[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; i++)
        out_batch_strides[i] = out_eff_strides[i];
    int out_stride_i = out_eff_strides[batch_ndim];
    int out_stride_j = out_eff_strides[batch_ndim + 1];

    /* -------------------- 优化分支：检查矩阵部分是否连续 -------------------- */
    int a_contig = is_contiguous_matrix_part(a, batch_ndim);
    int b_contig = is_contiguous_matrix_part(b, batch_ndim);
    int out_contig = is_contiguous_matrix_part(out, batch_ndim);

    if (a_contig && b_contig && out_contig)
    {
        // 所有矩阵子块连续，可使用分块乘法
        // 首先将输出数据清零（因为分块乘法是累加，而我们需要直接赋值）
        memset(out->data, 0, out->size * sizeof(float));

        int batch_coords[TENSOR_MAX_DIM] = {0};
        while (1)
        {
            size_t a_base = 0, b_base = 0, out_base = 0;
            for (int d = 0; d < batch_ndim; d++)
            {
                a_base += batch_coords[d] * a_batch_strides[d];
                b_base += batch_coords[d] * b_batch_strides[d];
                out_base += batch_coords[d] * out_batch_strides[d];
            }

            matmul_tiled(a_M, b_N, a_K,
                         a->data + a_base,
                         b->data + b_base,
                         out->data + out_base);

            if (util_increment_coords(batch_coords, batch_dims, batch_ndim))
                break;
        }
        return TENSOR_OK;
    }
    /* -------------------- 回退到通用循环 -------------------- */
    else
    {
        int batch_coords[TENSOR_MAX_DIM] = {0};
        while (1)
        {
            size_t a_base = 0, b_base = 0, out_base = 0;
            for (int d = 0; d < batch_ndim; d++)
            {
                a_base += batch_coords[d] * a_batch_strides[d];
                b_base += batch_coords[d] * b_batch_strides[d];
                out_base += batch_coords[d] * out_batch_strides[d];
            }

            for (int i = 0; i < a_M; i++)
            {
                for (int j = 0; j < b_N; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < a_K; k++)
                    {
                        float av = a->data[a_base + i * a_stride_i + k * a_stride_k];
                        float bv = b->data[b_base + k * b_stride_k + j * b_stride_j];
                        sum += (double)av * bv;
                    }
                    size_t out_off = out_base + i * out_stride_i + j * out_stride_j;
                    out->data[out_off] = (float)sum;
                }
            }

            if (util_increment_coords(batch_coords, batch_dims, batch_ndim))
                break;
        }
        return TENSOR_OK;
    }
}

/* ==================== 批量矩阵乘法（严格三维） ==================== */

/**
 * @brief 批量矩阵乘法（严格三维）
 * @param a   输入张量 A，形状 [batch, M, K]
 * @param b   输入张量 B，形状 [batch, K, N]
 * @param out 输出张量，形状 [batch, M, N]
 * @return TensorStatus
 */
TensorStatus tensor_bmm(const Tensor *a, const Tensor *b, Tensor *out)
{
    if (!a || !b || !out)
        return TENSOR_ERR_NULL_PTR;
    if (a->ndim != 3 || b->ndim != 3)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (a->dims[0] != b->dims[0])
        return TENSOR_ERR_SHAPE_MISMATCH;

    int batch = a->dims[0];
    int M = a->dims[1];
    int K = a->dims[2];
    int K2 = b->dims[1];
    int N = b->dims[2];
    if (K != K2)
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (out->ndim != 3 || out->dims[0] != batch ||
        out->dims[1] != M || out->dims[2] != N)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int a_strides[3], b_strides[3], out_strides[3];
    util_get_effective_strides(a, a_strides);
    util_get_effective_strides(b, b_strides);
    util_get_effective_strides(out, out_strides);

    for (int n = 0; n < batch; n++)
    {
        size_t a_base = n * a_strides[0];
        size_t b_base = n * b_strides[0];
        size_t out_base = n * out_strides[0];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < K; k++)
                {
                    float av = a->data[a_base + i * a_strides[1] + k * a_strides[2]];
                    float bv = b->data[b_base + k * b_strides[1] + j * b_strides[2]];
                    sum += (double)av * bv;
                }
                out->data[out_base + i * out_strides[1] + j * out_strides[2]] = (float)sum;
            }
        }
    }
    return TENSOR_OK;
}

/* ==================== 向量点积 ==================== */

/**
 * @brief 向量点积（两个 1 维张量）
 * @param a   输入向量
 * @param b   输入向量
 * @param out 输出标量张量（0维或1维长度为1）
 * @return TensorStatus
 */
TensorStatus tensor_dot(const Tensor *a, const Tensor *b, Tensor *out)
{
    if (!a || !b || !out)
        return TENSOR_ERR_NULL_PTR;
    if (a->ndim != 1 || b->ndim != 1)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (a->size != b->size)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (out->ndim != 0 && !(out->ndim == 1 && out->dims[0] == 1))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    double sum = 0.0;
    for (size_t i = 0; i < a->size; i++)
        sum += (double)a->data[i] * b->data[i];
    out->data[0] = (float)sum;
    return TENSOR_OK;
}

/* ==================== 向量外积 ==================== */

/**
 * @brief 向量外积（两个 1 维张量）
 * @param a   输入向量 (m)
 * @param b   输入向量 (n)
 * @param out 输出2维张量 (m, n)
 * @return TensorStatus
 */
TensorStatus tensor_outer(const Tensor *a, const Tensor *b, Tensor *out)
{
    if (!a || !b || !out)
        return TENSOR_ERR_NULL_PTR;
    if (a->ndim != 1 || b->ndim != 1)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int m = (int)a->size;
    int n = (int)b->size;
    if (out->ndim != 2 || out->dims[0] != m || out->dims[1] != n)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int out_strides[2];
    util_get_effective_strides(out, out_strides);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            size_t off = i * out_strides[0] + j * out_strides[1];
            out->data[off] = a->data[i] * b->data[j];
        }
    }
    return TENSOR_OK;
}

/* ==================== 张量缩并（tensordot） ==================== */

/**
 * @brief 张量缩并（广义点积）
 * @param a       输入张量 A
 * @param b       输入张量 B
 * @param axes_a  要缩并的 A 的轴索引数组
 * @param axes_b  要缩并的 B 的轴索引数组
 * @param naxes   轴对数
 * @param out     输出张量
 * @return TensorStatus
 */
TensorStatus tensor_tensordot(const Tensor *a, const Tensor *b,
                              const int *axes_a, const int *axes_b, int naxes,
                              Tensor *out)
{
    if (!a || !b || !out || !axes_a || !axes_b)
        return TENSOR_ERR_NULL_PTR;
    if (naxes <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    int a_ndim = a->ndim, b_ndim = b->ndim;

    /* 验证轴并检查对应维度相等 */
    for (int i = 0; i < naxes; i++)
    {
        int ax_a = util_normalize_axis(axes_a[i], a_ndim);
        int ax_b = util_normalize_axis(axes_b[i], b_ndim);
        if (ax_a < 0 || ax_b < 0)
            return TENSOR_ERR_INVALID_PARAM;
        if (a->dims[ax_a] != b->dims[ax_b])
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    /* 标记 a 中哪些轴是缩并轴 */
    int a_is_reduce[TENSOR_MAX_DIM] = {0};
    for (int i = 0; i < naxes; i++)
    {
        int ax = util_normalize_axis(axes_a[i], a_ndim);
        a_is_reduce[ax] = 1;
    }
    int a_eff_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(a, a_eff_strides);
    int a_keep_dims[TENSOR_MAX_DIM], a_keep_strides[TENSOR_MAX_DIM];
    int a_reduce_dims[TENSOR_MAX_DIM], a_reduce_strides[TENSOR_MAX_DIM];
    int keep_idx = 0, red_idx = 0;
    for (int i = 0; i < a_ndim; i++)
    {
        if (a_is_reduce[i])
        {
            a_reduce_dims[red_idx] = a->dims[i];
            a_reduce_strides[red_idx] = a_eff_strides[i];
            red_idx++;
        }
        else
        {
            a_keep_dims[keep_idx] = a->dims[i];
            a_keep_strides[keep_idx] = a_eff_strides[i];
            keep_idx++;
        }
    }
    int a_keep_ndim = keep_idx;
    int a_reduce_ndim = red_idx;

    /* 标记 b 中哪些轴是缩并轴 */
    int b_is_reduce[TENSOR_MAX_DIM] = {0};
    for (int i = 0; i < naxes; i++)
    {
        int ax = util_normalize_axis(axes_b[i], b_ndim);
        b_is_reduce[ax] = 1;
    }
    int b_eff_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(b, b_eff_strides);
    int b_keep_dims[TENSOR_MAX_DIM], b_keep_strides[TENSOR_MAX_DIM];
    int b_reduce_dims[TENSOR_MAX_DIM], b_reduce_strides[TENSOR_MAX_DIM];
    keep_idx = 0;
    red_idx = 0;
    for (int i = 0; i < b_ndim; i++)
    {
        if (b_is_reduce[i])
        {
            b_reduce_dims[red_idx] = b->dims[i];
            b_reduce_strides[red_idx] = b_eff_strides[i];
            red_idx++;
        }
        else
        {
            b_keep_dims[keep_idx] = b->dims[i];
            b_keep_strides[keep_idx] = b_eff_strides[i];
            keep_idx++;
        }
    }
    int b_keep_ndim = keep_idx;
    if (red_idx != naxes)
        return TENSOR_ERR_INVALID_PARAM;

    /* 输出形状 = a_keep_dims + b_keep_dims */
    int out_ndim = a_keep_ndim + b_keep_ndim;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < a_keep_ndim; i++)
        out_dims[i] = a_keep_dims[i];
    for (int i = 0; i < b_keep_ndim; i++)
        out_dims[a_keep_ndim + i] = b_keep_dims[i];

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; i++)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int out_eff_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(out, out_eff_strides);

    /* 遍历输出坐标 */
    int out_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        /* 计算 a 和 b 的基偏移（仅由保留轴贡献） */
        size_t a_base = 0;
        for (int i = 0; i < a_keep_ndim; i++)
            a_base += out_coords[i] * a_keep_strides[i];
        size_t b_base = 0;
        for (int i = 0; i < b_keep_ndim; i++)
            b_base += out_coords[a_keep_ndim + i] * b_keep_strides[i];

        /* 遍历所有缩并轴的组合 */
        double sum = 0.0;
        int red_coords[TENSOR_MAX_DIM] = {0};
        while (1)
        {
            size_t a_red_off = 0, b_red_off = 0;
            for (int i = 0; i < naxes; i++)
            {
                a_red_off += red_coords[i] * a_reduce_strides[i];
                b_red_off += red_coords[i] * b_reduce_strides[i];
            }
            sum += (double)a->data[a_base + a_red_off] * b->data[b_base + b_red_off];

            if (util_increment_coords(red_coords, a_reduce_dims, naxes))
                break;
        }

        size_t out_off = 0;
        for (int i = 0; i < out_ndim; i++)
            out_off += out_coords[i] * out_eff_strides[i];
        out->data[out_off] = (float)sum;

        if (util_increment_coords(out_coords, out_dims, out_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ==================== 转置/形状操作 ==================== */

/**
 * @brief 矩阵转置（交换最后两维）
 * @param src 输入张量（至少2维）
 * @param out 输出张量（视图，数据复制）
 * @return TensorStatus
 */
TensorStatus tensor_transpose(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 2)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim = src->ndim;
    for (int i = 0; i < ndim - 2; i++)
        if (src->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;
    if (src->dims[ndim - 2] != out->dims[ndim - 1] ||
        src->dims[ndim - 1] != out->dims[ndim - 2])
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        int src_coords[TENSOR_MAX_DIM];
        memcpy(src_coords, coords, ndim * sizeof(int));
        src_coords[ndim - 2] = coords[ndim - 1];
        src_coords[ndim - 1] = coords[ndim - 2];

        size_t src_off = 0, out_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            src_off += src_coords[i] * src_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        out->data[out_off] = src->data[src_off];

        if (util_increment_coords(coords, out->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/**
 * @brief 一般转置（按指定顺序重排轴）
 * @param src  输入张量
 * @param axes 新轴的顺序，长度为 src 的维度数
 * @param out  输出张量（视图，数据复制）
 * @return TensorStatus
 */
TensorStatus tensor_permute(const Tensor *src, const int *axes, Tensor *out)
{
    if (!src || !out || !axes)
        return TENSOR_ERR_NULL_PTR;
    int ndim = src->ndim;
    int used[TENSOR_MAX_DIM] = {0};
    for (int i = 0; i < ndim; i++)
    {
        int ax = axes[i];
        if (ax < 0 || ax >= ndim || used[ax])
            return TENSOR_ERR_INVALID_PARAM;
        used[ax] = 1;
        if (out->dims[i] != src->dims[ax])
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        int src_coords[TENSOR_MAX_DIM];
        for (int i = 0; i < ndim; i++)
            src_coords[axes[i]] = coords[i];

        size_t src_off = 0, out_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            src_off += src_coords[i] * src_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        out->data[out_off] = src->data[src_off];

        if (util_increment_coords(coords, out->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/**
 * @brief 提取对角线（2维输入）或创建对角阵（1维输入）
 * @param src 输入张量
 * @param out 输出张量
 * @return TensorStatus
 */
TensorStatus tensor_diag(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;

    if (src->ndim == 1)
    {
        int n = (int)src->size;
        if (out->ndim != 2 || out->dims[0] != n || out->dims[1] != n)
            return TENSOR_ERR_SHAPE_MISMATCH;

        TensorStatus status = tensor_make_unique(out);
        if (status != TENSOR_OK)
            return status;

        memset(out->data, 0, out->size * sizeof(float));

        int out_strides[2];
        util_get_effective_strides(out, out_strides);
        for (int i = 0; i < n; i++)
        {
            size_t off = i * out_strides[0] + i * out_strides[1];
            out->data[off] = src->data[i];
        }
    }
    else if (src->ndim == 2)
    {
        int rows = src->dims[0];
        int cols = src->dims[1];
        int diag_len = (rows < cols) ? rows : cols;
        if (out->ndim != 1 || out->dims[0] != diag_len)
            return TENSOR_ERR_SHAPE_MISMATCH;

        TensorStatus status = tensor_make_unique(out);
        if (status != TENSOR_OK)
            return status;

        int src_strides[2];
        util_get_effective_strides(src, src_strides);
        int out_strides[1];
        util_get_effective_strides(out, out_strides);

        for (int i = 0; i < diag_len; i++)
        {
            size_t src_off = i * src_strides[0] + i * src_strides[1];
            out->data[i * out_strides[0]] = src->data[src_off];
        }
    }
    else
    {
        return TENSOR_ERR_INVALID_PARAM;
    }
    return TENSOR_OK;
}

/**
 * @brief 迹（沿 axis1 和 axis2 的对角线和）
 * @param src   输入张量
 * @param axis1 第一个轴
 * @param axis2 第二个轴
 * @param out   输出张量
 * @return TensorStatus
 */
TensorStatus tensor_trace(const Tensor *src, int axis1, int axis2, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    int ndim = src->ndim;
    int ax1 = util_normalize_axis(axis1, ndim);
    int ax2 = util_normalize_axis(axis2, ndim);
    if (ax1 < 0 || ax2 < 0 || ax1 == ax2)
        return TENSOR_ERR_INVALID_PARAM;
    if (src->dims[ax1] != src->dims[ax2])
        return TENSOR_ERR_SHAPE_MISMATCH;

    int out_ndim = ndim - 2;
    int out_dims[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; i++)
        if (i != ax1 && i != ax2)
            out_dims[idx++] = src->dims[i];

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; i++)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(out, out_strides);

    int out_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        int src_base[TENSOR_MAX_DIM];
        int out_idx = 0;
        for (int i = 0; i < ndim; i++)
        {
            if (i == ax1 || i == ax2)
                src_base[i] = 0;
            else
                src_base[i] = out_coords[out_idx++];
        }

        double sum = 0.0;
        int diag_len = src->dims[ax1];
        for (int k = 0; k < diag_len; k++)
        {
            src_base[ax1] = k;
            src_base[ax2] = k;
            size_t src_off = 0;
            for (int i = 0; i < ndim; i++)
                src_off += src_base[i] * src_strides[i];
            sum += src->data[src_off];
        }

        size_t out_off = 0;
        for (int i = 0; i < out_ndim; i++)
            out_off += out_coords[i] * out_strides[i];
        out->data[out_off] = (float)sum;
        if (util_increment_coords(out_coords, out_dims, out_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ==================== 矩阵分解与求解 ==================== */

/**
 * @brief 矩阵求逆（使用 LU 分解）
 * @param src 输入方阵（2维）
 * @param out 输出方阵
 * @return TensorStatus
 */
TensorStatus tensor_inv(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 2)
        return TENSOR_ERR_INVALID_PARAM;
    int n = src->dims[src->ndim - 1];
    if (src->dims[src->ndim - 2] != n)
        return TENSOR_ERR_INVALID_PARAM;
    if (out->ndim != src->ndim || !util_shapes_equal(out->dims, src->dims, src->ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(out);
    if (status != TENSOR_OK)
        return status;

    Tensor *cont_src = (Tensor *)src;
    int need_free = 0;
    if (!util_is_contiguous(src))
    {
        cont_src = tensor_clone(src);
        if (!cont_src)
            return TENSOR_ERR_MEMORY;
        need_free = 1;
    }

    int batch_ndim = src->ndim - 2;
    int batch_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; ++i)
        batch_dims[i] = src->dims[i];

    int src_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(cont_src, src_strides);
    util_get_effective_strides(out, out_strides);

    float *LU = (float *)malloc(n * n * sizeof(float));
    int *pivot = (int *)malloc(n * sizeof(int));
    float *b = (float *)malloc(n * sizeof(float));
    float *x = (float *)malloc(n * sizeof(float));
    if (!LU || !pivot || !b || !x)
    {
        free(LU);
        free(pivot);
        free(b);
        free(x);
        if (need_free)
            tensor_destroy(cont_src);
        return TENSOR_ERR_MEMORY;
    }

    int batch_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_base = 0;
        for (int i = 0; i < batch_ndim; ++i)
            src_base += batch_coords[i] * src_strides[i];

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                size_t off = src_base + i * src_strides[batch_ndim] + j * src_strides[batch_ndim + 1];
                LU[i * n + j] = cont_src->data[off];
            }
        }

        if (util_lu_decompose(LU, n, LU, pivot) != 0)
        {
            free(LU);
            free(pivot);
            free(b);
            free(x);
            if (need_free)
                tensor_destroy(cont_src);
            return TENSOR_ERR_DIV_BY_ZERO;
        }

        size_t out_base = 0;
        for (int i = 0; i < batch_ndim; ++i)
            out_base += batch_coords[i] * out_strides[i];

        for (int col = 0; col < n; ++col)
        {
            for (int i = 0; i < n; ++i)
                b[i] = (i == col) ? 1.0f : 0.0f;
            TensorStatus st = util_lu_solve(LU, n, pivot, b, x);
            if (st != TENSOR_OK)
            {
                free(LU);
                free(pivot);
                free(b);
                free(x);
                if (need_free)
                    tensor_destroy(cont_src);
                return st;
            }
            for (int i = 0; i < n; ++i)
            {
                size_t off = out_base + i * out_strides[batch_ndim] + col * out_strides[batch_ndim + 1];
                out->data[off] = x[i];
            }
        }

        if (util_increment_coords(batch_coords, batch_dims, batch_ndim))
            break;
    }

    free(LU);
    free(pivot);
    free(b);
    free(x);
    if (need_free)
        tensor_destroy(cont_src);
    return TENSOR_OK;
}

/**
 * @brief 计算方阵的行列式（使用 LU 分解）
 * @param src 输入方阵（2维）
 * @param out 输出标量张量（0维或1维长度为1）
 * @return TensorStatus
 */
TensorStatus tensor_det(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != 2)
        return TENSOR_ERR_INVALID_PARAM;
    int n = src->dims[0];
    if (src->dims[1] != n)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 0 && !(out->ndim == 1 && out->dims[0] == 1))
        return TENSOR_ERR_SHAPE_MISMATCH;

    float *A = (float *)malloc(n * n * sizeof(float));
    if (!A)
        return TENSOR_ERR_MEMORY;
    int src_strides[2];
    util_get_effective_strides(src, src_strides);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = src->data[i * src_strides[0] + j * src_strides[1]];
        }
    }

    float *LU = (float *)malloc(n * n * sizeof(float));
    int *pivot = (int *)malloc(n * sizeof(int));
    if (!LU || !pivot)
    {
        free(A);
        free(LU);
        free(pivot);
        return TENSOR_ERR_MEMORY;
    }
    int status = util_lu_decompose(A, n, LU, pivot);
    free(A);
    if (status != 0)
    {
        free(LU);
        free(pivot);
        tensor_make_unique(out);
        out->data[0] = 0.0f;
        return TENSOR_ERR_SINGULAR_MATRIX;
    }

    double det = 1.0;
    int sign = 1;
    for (int i = 0; i < n; ++i)
    {
        if (pivot[i] != i)
            sign = -sign;
        det *= LU[i * n + i];
    }
    det = sign * det;

    TensorStatus ts = tensor_make_unique(out);
    if (ts != TENSOR_OK)
    {
        free(LU);
        free(pivot);
        return ts;
    }
    out->data[0] = (float)det;

    free(LU);
    free(pivot);
    return TENSOR_OK;
}

/**
 * @brief 解线性方程组 AX = B
 * @param A 系数矩阵（2维方阵）
 * @param B 右侧张量（1维或2维，最后一维大小等于 n）
 * @param X 输出解张量，形状与 B 相同
 * @return TensorStatus
 */
TensorStatus tensor_solve(const Tensor *A, const Tensor *B, Tensor *X)
{
    if (!A || !B || !X)
        return TENSOR_ERR_NULL_PTR;
    if (A->ndim != 2)
        return TENSOR_ERR_INVALID_PARAM;
    int n = A->dims[0];
    if (A->dims[1] != n)
        return TENSOR_ERR_INVALID_PARAM;

    int b_ndim = B->ndim;
    int k = 1;
    if (b_ndim == 1)
    {
        if (B->dims[0] != n)
            return TENSOR_ERR_SHAPE_MISMATCH;
    }
    else if (b_ndim == 2)
    {
        if (B->dims[0] != n)
            return TENSOR_ERR_SHAPE_MISMATCH;
        k = B->dims[1];
    }
    else
    {
        return TENSOR_ERR_SHAPE_MISMATCH;
    }

    if (X->ndim != b_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < b_ndim; ++i)
        if (X->dims[i] != B->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    float *A_data = (float *)malloc(n * n * sizeof(float));
    if (!A_data)
        return TENSOR_ERR_MEMORY;
    int A_strides[2];
    util_get_effective_strides(A, A_strides);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A_data[i * n + j] = A->data[i * A_strides[0] + j * A_strides[1]];
        }
    }

    float *LU = (float *)malloc(n * n * sizeof(float));
    int *pivot = (int *)malloc(n * sizeof(int));
    if (!LU || !pivot)
    {
        free(A_data);
        free(LU);
        free(pivot);
        return TENSOR_ERR_MEMORY;
    }
    int status = util_lu_decompose(A_data, n, LU, pivot);
    free(A_data);
    if (status != 0)
    {
        free(LU);
        free(pivot);
        return TENSOR_ERR_DIV_BY_ZERO;
    }

    int B_strides[TENSOR_MAX_DIM], X_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(B, B_strides);
    util_get_effective_strides(X, X_strides);

    float *b_col = (float *)malloc(n * sizeof(float));
    float *x_col = (float *)malloc(n * sizeof(float));
    if (!b_col || !x_col)
    {
        free(LU);
        free(pivot);
        free(b_col);
        free(x_col);
        return TENSOR_ERR_MEMORY;
    }

    for (int col = 0; col < k; ++col)
    {
        for (int i = 0; i < n; ++i)
        {
            size_t off;
            if (b_ndim == 1)
                off = i * B_strides[0];
            else
                off = i * B_strides[0] + col * B_strides[1];
            b_col[i] = B->data[off];
        }

        TensorStatus st = util_lu_solve(LU, n, pivot, b_col, x_col);
        if (st != TENSOR_OK)
        {
            free(LU);
            free(pivot);
            free(b_col);
            free(x_col);
            return st;
        }

        tensor_make_unique(X);
        for (int i = 0; i < n; ++i)
        {
            size_t off;
            if (b_ndim == 1)
                off = i * X_strides[0];
            else
                off = i * X_strides[0] + col * X_strides[1];
            X->data[off] = x_col[i];
        }
    }

    free(LU);
    free(pivot);
    free(b_col);
    free(x_col);
    return TENSOR_OK;
}

/**
 * @brief Cholesky 分解：A = L * L^T（A 对称正定）
 * @param src 输入对称正定方阵
 * @param out 输出下三角矩阵 L
 * @return TensorStatus
 */
TensorStatus tensor_cholesky(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != 2)
        return TENSOR_ERR_INVALID_PARAM;
    int n = src->dims[0];
    if (src->dims[1] != n)
        return TENSOR_ERR_INVALID_PARAM;
    if (out->ndim != 2 || out->dims[0] != n || out->dims[1] != n)
        return TENSOR_ERR_SHAPE_MISMATCH;

    double *A = (double *)malloc(n * n * sizeof(double));
    if (!A)
        return TENSOR_ERR_MEMORY;
    int src_strides[2];
    util_get_effective_strides(src, src_strides);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = src->data[i * src_strides[0] + j * src_strides[1]];
        }
    }

    const double eps = 1e-12;
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            double diff = fabs(A[i * n + j] - A[j * n + i]);
            double max_val = fmax(fabs(A[i * n + j]), fabs(A[j * n + i]));
            if (diff > eps * (1.0 + max_val))
            {
                free(A);
                return TENSOR_ERR_INVALID_PARAM;
            }
        }
    }

    double *L = (double *)calloc(n * n, sizeof(double));
    if (!L)
    {
        free(A);
        return TENSOR_ERR_MEMORY;
    }

    for (int j = 0; j < n; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < j; ++k)
        {
            sum += L[j * n + k] * L[j * n + k];
        }
        double d = A[j * n + j] - sum;
        if (d <= 0.0)
        {
            free(A);
            free(L);
            return TENSOR_ERR_INVALID_PARAM;
        }
        L[j * n + j] = sqrt(d);

        for (int i = j + 1; i < n; ++i)
        {
            sum = 0.0;
            for (int k = 0; k < j; ++k)
            {
                sum += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
        }
    }
    free(A);

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
    {
        free(L);
        return status;
    }

    int out_strides[2];
    util_get_effective_strides(out, out_strides);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            size_t off = i * out_strides[0] + j * out_strides[1];
            out->data[off] = (float)L[i * n + j];
        }
        for (int j = i + 1; j < n; ++j)
        {
            size_t off = i * out_strides[0] + j * out_strides[1];
            out->data[off] = 0.0f;
        }
    }

    free(L);
    return TENSOR_OK;
}

/**
 * @brief 奇异值分解（双边 Jacobi 方法）
 * @param src  输入矩阵（2维）
 * @param U    输出左奇异矩阵
 * @param S    输出奇异值向量
 * @param V    输出右奇异矩阵
 * @param full 非零表示完整矩阵，否则经济分解
 * @return TensorStatus
 */
TensorStatus tensor_svd(const Tensor *src, Tensor *U, Tensor *S, Tensor *V, int full)
{
    if (!src || !U || !S || !V)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != 2)
        return TENSOR_ERR_INVALID_PARAM;
    int m = src->dims[0];
    int n = src->dims[1];
    int k = (m < n) ? m : n;

    if (full)
    {
        if (U->ndim != 2 || U->dims[0] != m || U->dims[1] != m)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (V->ndim != 2 || V->dims[0] != n || V->dims[1] != n)
            return TENSOR_ERR_SHAPE_MISMATCH;
    }
    else
    {
        if (U->ndim != 2 || U->dims[0] != m || U->dims[1] != k)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (V->ndim != 2 || V->dims[0] != n || V->dims[1] != k)
            return TENSOR_ERR_SHAPE_MISMATCH;
    }
    if (S->ndim != 1 || S->dims[0] != k)
        return TENSOR_ERR_SHAPE_MISMATCH;

    double *A = (double *)malloc(m * n * sizeof(double));
    if (!A)
        return TENSOR_ERR_MEMORY;
    int src_strides[2];
    util_get_effective_strides(src, src_strides);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = src->data[i * src_strides[0] + j * src_strides[1]];
        }
    }

    double *V_mat = (double *)calloc(n * n, sizeof(double));
    if (!V_mat)
    {
        free(A);
        return TENSOR_ERR_MEMORY;
    }
    for (int i = 0; i < n; ++i)
        V_mat[i * n + i] = 1.0;

    double *v = (double *)malloc(m * sizeof(double));
    if (!v)
    {
        free(A);
        free(V_mat);
        return TENSOR_ERR_MEMORY;
    }

    const int max_iter = 100 * n * n;
    const double tol = 1e-10;
    int iter = 0;
    while (iter < max_iter)
    {
        double max_val = 0.0;
        int p = -1, q = -1;
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                double dot = 0.0;
                for (int r = 0; r < m; ++r)
                    dot += A[r * n + i] * A[r * n + j];
                double norm_i = 0.0, norm_j = 0.0;
                for (int r = 0; r < m; ++r)
                {
                    norm_i += A[r * n + i] * A[r * n + i];
                    norm_j += A[r * n + j] * A[r * n + j];
                }
                double val = fabs(dot) / sqrt(norm_i * norm_j + 1e-15);
                if (val > max_val)
                {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_val < tol)
            break;

        double a = 0.0, b = 0.0, c = 0.0;
        for (int r = 0; r < m; ++r)
        {
            a += A[r * n + p] * A[r * n + p];
            b += A[r * n + q] * A[r * n + q];
            c += A[r * n + p] * A[r * n + q];
        }
        double tau = (b - a) / (2.0 * c + 1e-15);
        double t;
        if (tau >= 0)
            t = 1.0 / (tau + sqrt(1.0 + tau * tau));
        else
            t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
        double cos = 1.0 / sqrt(1.0 + t * t);
        double sin = t * cos;

        for (int r = 0; r < m; ++r)
        {
            double a_rp = A[r * n + p];
            double a_rq = A[r * n + q];
            A[r * n + p] = a_rp * cos - a_rq * sin;
            A[r * n + q] = a_rp * sin + a_rq * cos;
        }

        for (int r = 0; r < n; ++r)
        {
            double v_rp = V_mat[r * n + p];
            double v_rq = V_mat[r * n + q];
            V_mat[r * n + p] = v_rp * cos - v_rq * sin;
            V_mat[r * n + q] = v_rp * sin + v_rq * cos;
        }
        ++iter;
    }

    double *sigma = (double *)malloc(k * sizeof(double));
    if (!sigma)
    {
        free(A);
        free(V_mat);
        free(v);
        return TENSOR_ERR_MEMORY;
    }
    for (int j = 0; j < k; ++j)
    {
        double norm = 0.0;
        for (int i = 0; i < m; ++i)
            norm += A[i * n + j] * A[i * n + j];
        sigma[j] = sqrt(norm);
    }

    // 计算有效秩（非零奇异值个数）
    double max_sigma = 0.0;
    for (int j = 0; j < k; ++j)
        if (sigma[j] > max_sigma)
            max_sigma = sigma[j];
    double rank_tol = 1e-12 * max_sigma * (m > n ? m : n);
    int *idx_nonzero = (int *)malloc(k * sizeof(int));
    int r = 0;
    for (int j = 0; j < k; ++j)
    {
        if (sigma[j] > rank_tol)
            idx_nonzero[r++] = j;
    }

    // 对非零奇异值按降序排序（简单冒泡）
    for (int i = 0; i < r - 1; ++i)
    {
        for (int j = i + 1; j < r; ++j)
        {
            if (sigma[idx_nonzero[i]] < sigma[idx_nonzero[j]])
            {
                int tmp = idx_nonzero[i];
                idx_nonzero[i] = idx_nonzero[j];
                idx_nonzero[j] = tmp;
            }
        }
    }

    int total_cols_U = full ? m : k;
    int total_cols_V = full ? n : k;

    double *U_mat = (double *)calloc(m * total_cols_U, sizeof(double));
    double *V_out = (double *)calloc(n * total_cols_V, sizeof(double));
    if (!U_mat || !V_out)
    {
        free(A);
        free(V_mat);
        free(sigma);
        free(v);
        free(idx_nonzero);
        free(U_mat);
        free(V_out);
        return TENSOR_ERR_MEMORY;
    }

    // 填充非零奇异值对应的列
    for (int pos = 0; pos < r; ++pos)
    {
        int j = idx_nonzero[pos];
        double inv_sigma = 1.0 / sigma[j];
        // U 列
        for (int i = 0; i < m; ++i)
        {
            U_mat[i * total_cols_U + pos] = A[i * n + j] * inv_sigma;
        }
        // V 列
        for (int i = 0; i < n; ++i)
        {
            V_out[i * total_cols_V + pos] = V_mat[i * n + j];
        }
    }

    // 填充剩余列（零奇异值空间）
    double *work = (double *)malloc(m * sizeof(double));
    if (!work)
    {
        free(A);
        free(V_mat);
        free(sigma);
        free(v);
        free(idx_nonzero);
        free(U_mat);
        free(V_out);
        return TENSOR_ERR_MEMORY;
    }

    for (int pos = r; pos < total_cols_U; ++pos)
    {
        int success = 0;
        for (int attempt = 0; attempt < 10; ++attempt)
        {
            util_random_double_vector(work, m);
            for (int j = 0; j < pos; ++j)
            {
                double dot = 0.0;
                for (int i = 0; i < m; ++i)
                    dot += U_mat[i * total_cols_U + j] * work[i];
                for (int i = 0; i < m; ++i)
                    work[i] -= dot * U_mat[i * total_cols_U + j];
            }
            double norm = 0.0;
            for (int i = 0; i < m; ++i)
                norm += work[i] * work[i];
            if (norm > rank_tol * rank_tol)
            {
                norm = sqrt(norm);
                for (int i = 0; i < m; ++i)
                    work[i] /= norm;
                success = 1;
                break;
            }
        }
        if (!success)
        {
            // 尝试标准基
            for (int base = 0; base < m; ++base)
            {
                for (int i = 0; i < m; ++i)
                    work[i] = (i == base) ? 1.0 : 0.0;
                for (int j = 0; j < pos; ++j)
                {
                    double dot = 0.0;
                    for (int i = 0; i < m; ++i)
                        dot += U_mat[i * total_cols_U + j] * work[i];
                    for (int i = 0; i < m; ++i)
                        work[i] -= dot * U_mat[i * total_cols_U + j];
                }
                double norm = 0.0;
                for (int i = 0; i < m; ++i)
                    norm += work[i] * work[i];
                if (norm > rank_tol * rank_tol)
                {
                    norm = sqrt(norm);
                    for (int i = 0; i < m; ++i)
                        work[i] /= norm;
                    success = 1;
                    break;
                }
            }
        }
        if (!success)
        {
            // 最后的手段：取第一个标准基（确保非零）
            for (int i = 0; i < m; ++i)
                work[i] = (i == 0) ? 1.0 : 0.0;
        }
        for (int i = 0; i < m; ++i)
            U_mat[i * total_cols_U + pos] = work[i];
    }

    // 对 V 填充剩余列（如果 full 且需要）
    if (full)
    {
        // 填充 V 的剩余列
        for (int pos = r; pos < total_cols_V; ++pos)
        {
            int success = 0;
            for (int attempt = 0; attempt < 10; ++attempt)
            {
                util_random_double_vector(work, n);
                for (int j = 0; j < pos; ++j)
                {
                    double dot = 0.0;
                    for (int i = 0; i < n; ++i)
                        dot += V_out[i * total_cols_V + j] * work[i];
                    for (int i = 0; i < n; ++i)
                        work[i] -= dot * V_out[i * total_cols_V + j];
                }
                double norm = 0.0;
                for (int i = 0; i < n; ++i)
                    norm += work[i] * work[i];
                if (norm > rank_tol * rank_tol)
                {
                    norm = sqrt(norm);
                    for (int i = 0; i < n; ++i)
                        work[i] /= norm;
                    success = 1;
                    break;
                }
            }
            if (!success)
            {
                for (int base = 0; base < n; ++base)
                {
                    for (int i = 0; i < n; ++i)
                        work[i] = (i == base) ? 1.0 : 0.0;
                    for (int j = 0; j < pos; ++j)
                    {
                        double dot = 0.0;
                        for (int i = 0; i < n; ++i)
                            dot += V_out[i * total_cols_V + j] * work[i];
                        for (int i = 0; i < n; ++i)
                            work[i] -= dot * V_out[i * total_cols_V + j];
                    }
                    double norm = 0.0;
                    for (int i = 0; i < n; ++i)
                        norm += work[i] * work[i];
                    if (norm > rank_tol * rank_tol)
                    {
                        norm = sqrt(norm);
                        for (int i = 0; i < n; ++i)
                            work[i] /= norm;
                        success = 1;
                        break;
                    }
                }
            }
            if (!success)
            {
                for (int i = 0; i < n; ++i)
                    work[i] = (i == pos % n) ? 1.0 : 0.0;
            }
            for (int i = 0; i < n; ++i)
                V_out[i * total_cols_V + pos] = work[i];
        }
    }

    // 写回 Tensor
    TensorStatus status;

    // 奇异值
    status = tensor_make_unique(S);
    if (status != TENSOR_OK)
        goto cleanup;
    int S_strides[1];
    util_get_effective_strides(S, S_strides);
    for (int j = 0; j < k; ++j)
    {
        // 注意：sigma 已排序，前 r 个是降序的非零值，后 k-r 个是零
        // 但在经济分解中，我们只输出前 k 个奇异值，按原顺序（已排序）
        // 这里我们按索引顺序赋值，但实际排序后的 sigma 顺序变了，需要重新映射
        // 简单起见，我们保持 sigma 的原始顺序？但为了与 U/V 匹配，U/V 的列已按排序后的顺序重排，
        // 所以 sigma 也应该按相同顺序输出。因此我们按排序后的顺序填充 sigma 数组。
        // 但 sigma 数组原本是按列索引的，现在我们只保存了排序后的顺序在 idx_nonzero 中，
        // 我们需要将排序后的值按顺序写入 S。
    }
    // 重新填充 sigma_sorted 数组
    double *sigma_sorted = (double *)calloc(k, sizeof(double));
    for (int pos = 0; pos < r; ++pos)
        sigma_sorted[pos] = sigma[idx_nonzero[pos]];
    // 后 k-r 个保持 0
    for (int j = 0; j < k; ++j)
        S->data[j * S_strides[0]] = (float)sigma_sorted[j];
    free(sigma_sorted);

    // U
    status = tensor_make_unique(U);
    if (status != TENSOR_OK)
        goto cleanup;
    int U_strides[2];
    util_get_effective_strides(U, U_strides);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < total_cols_U; ++j)
        {
            size_t off = i * U_strides[0] + j * U_strides[1];
            U->data[off] = (float)U_mat[i * total_cols_U + j];
        }

    // V
    status = tensor_make_unique(V);
    if (status != TENSOR_OK)
        goto cleanup;
    int V_strides[2];
    util_get_effective_strides(V, V_strides);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < total_cols_V; ++j)
        {
            size_t off = i * V_strides[0] + j * V_strides[1];
            V->data[off] = (float)V_out[i * total_cols_V + j];
        }

cleanup:
    free(A);
    free(V_mat);
    free(sigma);
    free(U_mat);
    free(V_out);
    free(v);
    free(work);
    free(idx_nonzero);
    return status;
}

/**
 * @brief QR 分解（修正 Gram-Schmidt）
 * @param src     输入矩阵（2维）
 * @param Q       输出正交矩阵
 * @param R       输出上三角矩阵
 * @param reduced 非零表示经济分解，否则完全分解
 * @return TensorStatus
 */
TensorStatus tensor_qr(const Tensor *src, Tensor *Q, Tensor *R, int reduced)
{
    if (!src || !Q || !R)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != 2)
        return TENSOR_ERR_INVALID_PARAM;

    int m = src->dims[0];
    int n = src->dims[1];
    int k = (m < n) ? m : n;

    if (reduced)
    {
        if (Q->ndim != 2 || Q->dims[0] != m || Q->dims[1] != k)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (R->ndim != 2 || R->dims[0] != k || R->dims[1] != n)
            return TENSOR_ERR_SHAPE_MISMATCH;
    }
    else
    {
        if (Q->ndim != 2 || Q->dims[0] != m || Q->dims[1] != m)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (R->ndim != 2 || R->dims[0] != m || R->dims[1] != n)
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    double normA = 0.0;
    {
        int src_strides[2];
        util_get_effective_strides(src, src_strides);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
            {
                float v = src->data[i * src_strides[0] + j * src_strides[1]];
                normA += (double)v * v;
            }
        normA = sqrt(normA);
    }
    const double eps = 1e-12;
    const double tol = (normA < eps) ? eps : normA * eps;

    double *A = (double *)malloc(m * n * sizeof(double));
    if (!A)
        return TENSOR_ERR_MEMORY;
    int src_strides[2];
    util_get_effective_strides(src, src_strides);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[j * m + i] = src->data[i * src_strides[0] + j * src_strides[1]];

    int q_cols = reduced ? k : m;
    int r_rows = reduced ? k : m;
    double *Q_mat = (double *)calloc(m * q_cols, sizeof(double));
    double *R_mat = (double *)calloc(r_rows * n, sizeof(double));
    double *v = (double *)malloc(m * sizeof(double));
    if (!Q_mat || !R_mat || !v)
    {
        free(A);
        free(Q_mat);
        free(R_mat);
        free(v);
        return TENSOR_ERR_MEMORY;
    }

    int r = 0;
    for (int j = 0; j < n && r < k; ++j)
    {
        double *a = A + j * m;
        memcpy(v, a, m * sizeof(double));

        for (int i = 0; i < r; ++i)
        {
            double dot = 0.0;
            for (int t = 0; t < m; ++t)
                dot += Q_mat[i * m + t] * v[t];
            R_mat[i * n + j] = dot;
            for (int t = 0; t < m; ++t)
                v[t] -= dot * Q_mat[i * m + t];
        }

        double norm_v = 0.0;
        for (int t = 0; t < m; ++t)
            norm_v += v[t] * v[t];
        norm_v = sqrt(norm_v);

        if (norm_v > tol)
        {
            double inv_norm = 1.0 / norm_v;
            for (int t = 0; t < m; ++t)
                Q_mat[r * m + t] = v[t] * inv_norm;
            R_mat[r * n + j] = norm_v;
            r++;
        }
        else
        {
            if (r < q_cols)
            {
                if (!find_orthogonal_vector(Q_mat, m, r, v, tol))
                {
                    free(A);
                    free(Q_mat);
                    free(R_mat);
                    free(v);
                    return TENSOR_ERR_INVALID_PARAM;
                }
                for (int t = 0; t < m; ++t)
                    Q_mat[r * m + t] = v[t];
                r++;
            }
        }
    }

    if (!reduced)
    {
        while (r < m)
        {
            if (!find_orthogonal_vector(Q_mat, m, r, v, tol))
            {
                free(A);
                free(Q_mat);
                free(R_mat);
                free(v);
                return TENSOR_ERR_INVALID_PARAM;
            }
            for (int t = 0; t < m; ++t)
                Q_mat[r * m + t] = v[t];
            r++;
        }
    }

    TensorStatus status;

    status = tensor_make_unique(Q);
    if (status != TENSOR_OK)
        goto cleanup;
    int Q_strides[2];
    util_get_effective_strides(Q, Q_strides);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < q_cols; ++j)
        {
            size_t off = i * Q_strides[0] + j * Q_strides[1];
            Q->data[off] = (float)Q_mat[j * m + i];
        }

    status = tensor_make_unique(R);
    if (status != TENSOR_OK)
        goto cleanup;
    int R_strides[2];
    util_get_effective_strides(R, R_strides);
    for (int i = 0; i < r_rows; ++i)
        for (int j = 0; j < n; ++j)
        {
            size_t off = i * R_strides[0] + j * R_strides[1];
            R->data[off] = (i <= j) ? (float)R_mat[i * n + j] : 0.0f;
        }

cleanup:
    free(A);
    free(Q_mat);
    free(R_mat);
    free(v);
    return status;
}

/**
 * @brief 计算实对称矩阵的特征值和特征向量（雅可比方法）
 * @param src      输入实对称方阵（2维）
 * @param eigvals  输出特征值张量（1维）
 * @param eigvecs  输出特征向量张量（2维，列为特征向量）
 * @return TensorStatus
 */
TensorStatus tensor_eigh(const Tensor *src, Tensor *eigvals, Tensor *eigvecs)
{
    if (!src || !eigvals || !eigvecs)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != 2)
        return TENSOR_ERR_INVALID_PARAM;
    int n = src->dims[0];
    if (src->dims[1] != n)
        return TENSOR_ERR_INVALID_PARAM;

    int src_strides[2];
    util_get_effective_strides(src, src_strides);
    const float *data = src->data;
    const float atol = 1e-5f;
    const float rtol = 1e-5f;
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            size_t off_ij = i * src_strides[0] + j * src_strides[1];
            size_t off_ji = j * src_strides[0] + i * src_strides[1];
            float a = data[off_ij];
            float b = data[off_ji];
            float diff = fabsf(a - b);
            float max_abs = fmaxf(fabsf(a), fabsf(b));
            if (diff > atol + rtol * max_abs)
                return TENSOR_ERR_INVALID_PARAM;
        }
    }

    if (eigvals->ndim != 1 || eigvals->dims[0] != n)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (eigvecs->ndim != 2 || eigvecs->dims[0] != n || eigvecs->dims[1] != n)
        return TENSOR_ERR_SHAPE_MISMATCH;

    double *A = (double *)malloc(n * n * sizeof(double));
    if (!A)
        return TENSOR_ERR_MEMORY;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            size_t off = i * src_strides[0] + j * src_strides[1];
            A[i * n + j] = src->data[off];
        }

    double *V = (double *)calloc(n * n, sizeof(double));
    if (!V)
    {
        free(A);
        return TENSOR_ERR_MEMORY;
    }
    for (int i = 0; i < n; ++i)
        V[i * n + i] = 1.0;

    const int max_iter = 100 * n * n;
    const double tol = 1e-10;
    int iter = 0;
    while (iter < max_iter)
    {
        double max_off = 0.0;
        int p = -1, q = -1;
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                double val = fabs(A[i * n + j]);
                if (val > max_off)
                {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < tol)
            break;

        double Apq = A[p * n + q];
        double App = A[p * n + p];
        double Aqq = A[q * n + q];
        double tau = (Aqq - App) / (2.0 * Apq);
        double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                              : -1.0 / (-tau + sqrt(1.0 + tau * tau));
        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;

        for (int r = 0; r < n; ++r)
        {
            double Apr = A[p * n + r];
            double Aqr = A[q * n + r];
            A[p * n + r] = c * Apr - s * Aqr;
            A[q * n + r] = s * Apr + c * Aqr;
        }
        for (int r = 0; r < n; ++r)
        {
            double Arp = A[r * n + p];
            double Arq = A[r * n + q];
            A[r * n + p] = c * Arp - s * Arq;
            A[r * n + q] = s * Arp + c * Arq;
        }

        for (int r = 0; r < n; ++r)
        {
            double Vrp = V[r * n + p];
            double Vrq = V[r * n + q];
            V[r * n + p] = c * Vrp - s * Vrq;
            V[r * n + q] = s * Vrp + c * Vrq;
        }
        ++iter;
    }

    TensorStatus status;

    status = tensor_make_unique(eigvals);
    if (status != TENSOR_OK)
    {
        free(A);
        free(V);
        return status;
    }
    int eig_strides[1];
    util_get_effective_strides(eigvals, eig_strides);
    for (int i = 0; i < n; ++i)
        eigvals->data[i * eig_strides[0]] = (float)A[i * n + i];

    status = tensor_make_unique(eigvecs);
    if (status != TENSOR_OK)
    {
        free(A);
        free(V);
        return status;
    }
    int vec_strides[2];
    util_get_effective_strides(eigvecs, vec_strides);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            size_t off = i * vec_strides[0] + j * vec_strides[1];
            eigvecs->data[off] = (float)V[i * n + j];
        }

    free(A);
    free(V);
    return TENSOR_OK;
}






/* ---------- 特征值分解辅助函数 ---------- */

/* 平衡矩阵（使行和列范数相近） */
static void balance(double *A, int n, int *low, int *high, double *scale)
{
    const double radix = 2.0;
    const double sqrdx = radix * radix;
    int i, j;
    int last = 0;
    for (i = 0; i < n; i++)
        scale[i] = 1.0;
    *low = 0;
    *high = n - 1;
    while (!last)
    {
        last = 1;
        for (i = *low; i <= *high; i++)
        {
            double r = 0.0, c = 0.0;
            for (j = *low; j <= *high; j++)
            {
                if (j != i)
                {
                    c += fabs(A[j * n + i]);
                    r += fabs(A[i * n + j]);
                }
            }
            if (c == 0.0 || r == 0.0)
            {
                if (i == *low)
                    (*low)++;
                else if (i == *high)
                    (*high)--;
                else
                {
                    // 交换行和列到边界
                    for (j = *low; j <= *high; j++)
                    {
                        double tmp = A[i * n + j];
                        A[i * n + j] = A[*high * n + j];
                        A[*high * n + j] = tmp;
                    }
                    for (j = *low; j <= *high; j++)
                    {
                        double tmp = A[j * n + i];
                        A[j * n + i] = A[j * n + *high];
                        A[j * n + *high] = tmp;
                    }
                    (*high)--;
                }
                last = 0;
            }
        }
    }
    // 缩放
    for (i = *low; i <= *high; i++)
    {
        double r = 0.0, c = 0.0;
        for (j = *low; j <= *high; j++)
        {
            if (j != i)
            {
                c += fabs(A[j * n + i]);
                r += fabs(A[i * n + j]);
            }
        }
        double g = r / radix;
        double f = 1.0;
        double s = c + r;
        while (c < g)
        {
            f *= radix;
            c *= sqrdx;
        }
        g = r * radix;
        while (c > g)
        {
            f /= radix;
            c /= sqrdx;
        }
        if ((c + r) / f < 0.95 * s)
        {
            scale[i] = f;
            for (j = *low; j <= *high; j++)
                A[i * n + j] /= f;
            for (j = *low; j <= *high; j++)
                A[j * n + i] *= f;
        }
    }
}

/* Hessenberg 约化（实矩阵） */
static void hessenberg(double *A, int n)
{
    for (int m = 1; m < n - 1; m++)
    {
        double x = 0.0;
        int i;
        for (i = m + 1; i < n; i++)
            x += fabs(A[i * n + m - 1]);
        if (x == 0.0)
            continue;
        double scale = 0.0;
        for (i = m; i < n; i++)
            scale += fabs(A[i * n + m - 1]);
        double u[n - m];
        double sum = 0.0;
        for (i = m; i < n; i++)
        {
            u[i - m] = A[i * n + m - 1] / scale;
            sum += u[i - m] * u[i - m];
        }
        double g = (u[0] >= 0) ? -sqrt(sum) : sqrt(sum);
        double h = u[0] * g - sum;
        u[0] -= g;
        // 行变换
        for (int j = m; j < n; j++)
        {
            double s = 0.0;
            for (int k = m; k < n; k++)
                s += u[k - m] * A[k * n + j];
            s /= h;
            for (int k = m; k < n; k++)
                A[k * n + j] -= s * u[k - m];
        }
        // 列变换
        for (int i2 = 0; i2 < n; i2++)
        {
            double s = 0.0;
            for (int k = m; k < n; k++)
                s += u[k - m] * A[i2 * n + k];
            s /= h;
            for (int k = m; k < n; k++)
                A[i2 * n + k] -= s * u[k - m];
        }
        A[(m + 1) * n + m - 1] = -g * scale;
        for (int i2 = m + 2; i2 < n; i2++)
            A[i2 * n + m - 1] = 0.0;
    }
}

/* 计算一个 2x2 块的特征值（用于双步 QR 的移位） */
static void hqr2_eigenvalues(double a, double b, double c, double d,
                             double *wr1, double *wi1, double *wr2, double *wi2)
{
    double tr = a + d;
    double det = (a * d - b * c);
    double disc = tr * tr - 4.0 * det;
    if (disc >= 0)
    {
        double sqrt_disc = sqrt(disc);
        *wr1 = (tr + sqrt_disc) / 2.0;
        *wr2 = (tr - sqrt_disc) / 2.0;
        *wi1 = *wi2 = 0.0;
    }
    else
    {
        *wr1 = *wr2 = tr / 2.0;
        *wi1 = sqrt(-disc) / 2.0;
        *wi2 = -*wi1;
    }
}

/* 双步 QR 迭代（计算上 Hessenberg 矩阵的全部特征值） */
static int hqr2(double *H, int n, double *wr, double *wi, double *Z)
{
    const double eps = 1e-12;
    int nn = n;
    int m, l, k, j, its, i;
    double t, x, y, w, p, q, r, s, z, u, v, a, b, c, d;
    while (nn > 0)
    {
        its = 0;
        do
        {
            for (l = nn - 1; l > 0; l--)
            {
                s = fabs(H[(l - 1) * n + l - 1]) + fabs(H[l * n + l]);
                if (s == 0.0)
                    s = 1.0;
                if (fabs(H[l * n + l - 1]) < eps * s)
                {
                    H[l * n + l - 1] = 0.0;
                    break;
                }
            }
            x = H[(nn - 1) * n + nn - 1];
            if (l == nn - 1)
            {
                // 单个实特征值
                wr[nn - 1] = H[(nn - 1) * n + nn - 1];
                wi[nn - 1] = 0.0;
                if (Z)
                {
                    // 特征向量暂不计算，可后续反迭代
                }
                nn--;
                break;
            }
            else if (l == nn - 2)
            {
                // 2x2 块，可能复特征值
                a = H[(nn - 2) * n + nn - 2];
                b = H[(nn - 2) * n + nn - 1];
                c = H[(nn - 1) * n + nn - 2];
                d = H[(nn - 1) * n + nn - 1];
                hqr2_eigenvalues(a, b, c, d, &wr[nn - 2], &wi[nn - 2], &wr[nn - 1], &wi[nn - 1]);
                if (Z)
                {
                    // 特征向量暂不处理
                }
                nn -= 2;
                break;
            }
            else
            {
                // 双步移位
                double y = H[(nn - 2) * n + nn - 2];
                double w = H[(nn - 1) * n + nn - 1];
                double s = H[(nn - 2) * n + nn - 1];
                double t = H[(nn - 1) * n + nn - 2];
                a = (y + w) / 2.0;
                b = (y - w) / 2.0;
                c = b * b + s * t;
                if (c >= 0)
                {
                    double sqrtc = sqrt(c);
                    x = a + sqrtc;
                    y = a - sqrtc;
                    if (fabs(x - w) < fabs(y - w))
                        x = y;
                }
                else
                {
                    x = a;
                }
                for (i = 0; i <= nn - 2; i++)
                {
                    H[i * n + i] -= x;
                }
                double q = H[(l + 1) * n + l];
                r = H[(l + 2) * n + l];
                for (k = l; k <= nn - 2; k++)
                {
                    if (k != l)
                    {
                        p = H[k * n + k - 1];
                        q = H[(k + 1) * n + k - 1];
                        r = (k + 2 <= nn - 1) ? H[(k + 2) * n + k - 1] : 0.0;
                    }
                    x = fabs(p) + fabs(q) + fabs(r);
                    if (x != 0.0)
                    {
                        p /= x;
                        q /= x;
                        r /= x;
                    }
                    s = sqrt(p * p + q * q + r * r);
                    if (p < 0)
                        s = -s;
                    if (s != 0.0)
                    {
                        if (k != l)
                            H[k * n + k - 1] = -s * x;
                        else if (l != 0)
                            H[k * n + k - 1] = -H[k * n + k - 1];
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;
                        // 行变换
                        for (j = k; j <= nn - 1; j++)
                        {
                            p = H[k * n + j] + q * H[(k + 1) * n + j];
                            if (k + 2 <= nn - 1)
                            {
                                p += r * H[(k + 2) * n + j];
                                H[(k + 2) * n + j] -= p * z;
                            }
                            H[(k + 1) * n + j] -= p * y;
                            H[k * n + j] -= p * x;
                        }
                        // 列变换
                        int jend = (k + 3 < nn) ? k + 3 : nn;
                        for (i = l; i <= jend; i++)
                        {
                            p = x * H[i * n + k] + y * H[i * n + k + 1];
                            if (k + 2 <= nn - 1)
                            {
                                p += z * H[i * n + k + 2];
                                H[i * n + k + 2] -= p * r;
                            }
                            H[i * n + k + 1] -= p * q;
                            H[i * n + k] -= p;
                        }
                    }
                }
                for (i = 0; i <= nn - 2; i++)
                {
                    H[i * n + i] += x;
                }
                its++;
            }
        } while (its < 30);
        if (its >= 30)
            return -1; // 不收敛
    }
    return 0;
}

/* 平衡逆缩放，恢复特征向量（若有） */
static void balback(double *V, int n, int low, int high, double *scale)
{
    for (int i = low; i <= high; i++)
    {
        double s = scale[i];
        for (int j = 0; j < n; j++)
            V[i * n + j] *= s;
        for (int j = 0; j < n; j++)
            V[j * n + i] /= s;
    }
}

/* 主特征值分解函数（仅计算特征值，不计算特征向量） */
static int eig_vals_only(const double *A, int n, double *wr, double *wi)
{
    double *H = (double *)malloc(n * n * sizeof(double));
    if (!H)
        return -1;
    for (int i = 0; i < n * n; i++)
        H[i] = A[i];
    int low, high;
    double *scale = (double *)malloc(n * sizeof(double));
    if (!scale)
    {
        free(H);
        return -1;
    }
    balance(H, n, &low, &high, scale);
    hessenberg(H, n);
    int status = hqr2(H, n, wr, wi, NULL);
    free(H);
    free(scale);
    return status;
}

/* 计算特征值和特征向量（使用反迭代） */
static int eig_vectors(const double *A, int n, double *wr, double *wi, double *Vr, double *Vi)
{
    // 简化：先计算特征值，然后对每个特征值用反迭代求特征向量
    // 对于复特征值，需要解复数方程组，这里略去，实际应使用复杂方法
    // 我们暂时仅当特征值全为实数时计算实特征向量
    int all_real = 1;
    for (int i = 0; i < n; i++)
        if (wi[i] != 0.0)
            all_real = 0;
    if (!all_real)
        return -1; // 暂不支持复特征向量

    // 使用反迭代求实特征向量
    double *H = (double *)malloc(n * n * sizeof(double));
    double *work = (double *)malloc(n * sizeof(double));
    if (!H || !work)
    {
        free(H);
        free(work);
        return -1;
    }
    for (int i = 0; i < n * n; i++)
        H[i] = A[i];

    for (int k = 0; k < n; k++)
    {
        double lambda = wr[k];
        // 构造 (A - lambda I)
        double *M = (double *)malloc(n * n * sizeof(double));
        memcpy(M, H, n * n * sizeof(double));
        for (int i = 0; i < n; i++)
            M[i * n + i] -= lambda;

        // 使用 LU 分解解 M x = b，取随机 b 并迭代
        double *b = (double *)malloc(n * sizeof(double));
        double *x = (double *)malloc(n * sizeof(double));
        int *pivot = (int *)malloc(n * sizeof(int));
        if (!M || !b || !x || !pivot)
        {
            free(M);
            free(b);
            free(x);
            free(pivot);
            free(H);
            free(work);
            return -1;
        }
        // LU 分解
        int *ipiv = pivot;
        for (int i = 0; i < n; i++)
            ipiv[i] = i;
        for (int i = 0; i < n; i++)
        {
            double max = fabs(M[i * n + i]);
            int p = i;
            for (int j = i + 1; j < n; j++)
            {
                if (fabs(M[j * n + i]) > max)
                {
                    max = fabs(M[j * n + i]);
                    p = j;
                }
            }
            if (max < 1e-12)
            {
                // 奇异，特征向量可能不唯一
                free(M);
                free(b);
                free(x);
                free(pivot);
                free(H);
                free(work);
                return -1;
            }
            if (p != i)
            {
                // 交换行
                for (int j = 0; j < n; j++)
                {
                    double tmp = M[i * n + j];
                    M[i * n + j] = M[p * n + j];
                    M[p * n + j] = tmp;
                }
                int tmp = ipiv[i];
                ipiv[i] = ipiv[p];
                ipiv[p] = tmp;
            }
            double inv = 1.0 / M[i * n + i];
            for (int j = i + 1; j < n; j++)
            {
                double factor = M[j * n + i] * inv;
                M[j * n + i] = factor;
                for (int l = i + 1; l < n; l++)
                {
                    M[j * n + l] -= factor * M[i * n + l];
                }
            }
        }
        // 随机右端项
        for (int i = 0; i < n; i++)
            b[i] = (double)rand() / RAND_MAX;
        // 前代
        for (int i = 0; i < n; i++)
        {
            double sum = b[i];
            for (int j = 0; j < i; j++)
                sum -= M[i * n + j] * x[j];
            x[i] = sum;
        }
        // 回代
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = x[i];
            for (int j = i + 1; j < n; j++)
                sum -= M[i * n + j] * x[j];
            x[i] = sum / M[i * n + i];
        }
        // 归一化
        double norm = 0.0;
        for (int i = 0; i < n; i++)
            norm += x[i] * x[i];
        norm = sqrt(norm);
        for (int i = 0; i < n; i++)
            Vr[k * n + i] = x[i] / norm;

        free(M);
        free(b);
        free(x);
        free(pivot);
    }
    free(H);
    free(work);
    return 0;
}

TensorStatus tensor_eig(const Tensor *src,
                        Tensor *eigvals_real, Tensor *eigvals_imag,
                        Tensor *eigvecs_real, Tensor *eigvecs_imag)
{
    if (!src || !eigvals_real || !eigvals_imag || !eigvecs_real || !eigvecs_imag)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != 2 || src->dims[0] != src->dims[1])
        return TENSOR_ERR_INVALID_PARAM;
    int n = src->dims[0];

    if (eigvals_real->ndim != 1 || eigvals_real->dims[0] != n ||
        eigvals_imag->ndim != 1 || eigvals_imag->dims[0] != n)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (eigvecs_real->ndim != 2 || eigvecs_real->dims[0] != n || eigvecs_real->dims[1] != n ||
        eigvecs_imag->ndim != 2 || eigvecs_imag->dims[0] != n || eigvecs_imag->dims[1] != n)
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 复制到 double 数组
    double *A = (double *)malloc(n * n * sizeof(double));
    if (!A)
        return TENSOR_ERR_MEMORY;
    int src_strides[2];
    util_get_effective_strides(src, src_strides);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = src->data[i * src_strides[0] + j * src_strides[1]];

    double *wr = (double *)malloc(n * sizeof(double));
    double *wi = (double *)malloc(n * sizeof(double));
    double *Vr = (double *)malloc(n * n * sizeof(double));
    double *Vi = (double *)malloc(n * n * sizeof(double));
    if (!wr || !wi || !Vr || !Vi)
    {
        free(A);
        free(wr);
        free(wi);
        free(Vr);
        free(Vi);
        return TENSOR_ERR_MEMORY;
    }

    // 先计算特征值
    int status = eig_vals_only(A, n, wr, wi);
    if (status != 0)
    {
        free(A);
        free(wr);
        free(wi);
        free(Vr);
        free(Vi);
        return TENSOR_ERR_NOT_IMPLEMENTED; // 或不收敛
    }

    // 尝试计算特征向量（仅当所有特征值实数时）
    if (eig_vectors(A, n, wr, wi, Vr, Vi) != 0)
    {
        // 若失败，特征向量部分置零
        memset(Vr, 0, n * n * sizeof(double));
        memset(Vi, 0, n * n * sizeof(double));
    }

    // 写入输出张量
    TensorStatus ts;
    ts = tensor_make_unique(eigvals_real);
    if (ts != TENSOR_OK)
        goto cleanup;
    ts = tensor_make_unique(eigvals_imag);
    if (ts != TENSOR_OK)
        goto cleanup;
    ts = tensor_make_unique(eigvecs_real);
    if (ts != TENSOR_OK)
        goto cleanup;
    ts = tensor_make_unique(eigvecs_imag);
    if (ts != TENSOR_OK)
        goto cleanup;

    // 特征值步长
    int real_strides[1], imag_strides[1];
    util_get_effective_strides(eigvals_real, real_strides);
    util_get_effective_strides(eigvals_imag, imag_strides);
    for (int i = 0; i < n; i++)
    {
        eigvals_real->data[i * real_strides[0]] = (float)wr[i];
        eigvals_imag->data[i * imag_strides[0]] = (float)wi[i];
    }

    // 特征向量步长
    int vr_strides[2], vi_strides[2];
    util_get_effective_strides(eigvecs_real, vr_strides);
    util_get_effective_strides(eigvecs_imag, vi_strides);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            eigvecs_real->data[i * vr_strides[0] + j * vr_strides[1]] = (float)Vr[i * n + j];
            eigvecs_imag->data[i * vi_strides[0] + j * vi_strides[1]] = (float)Vi[i * n + j];
        }

cleanup:
    free(A);
    free(wr);
    free(wi);
    free(Vr);
    free(Vi);
    return ts;
}
TensorStatus tensor_lstsq(const Tensor *A, const Tensor *B, Tensor *X)
{
    if (!A || !B || !X) return TENSOR_ERR_NULL_PTR;
    if (A->ndim != 2) return TENSOR_ERR_INVALID_PARAM;
    int m = A->dims[0];
    int n = A->dims[1];

    // 解析 B 的形状，确定右端项的数量 k
    int b_ndim = B->ndim;
    int k = 1;
    if (b_ndim == 1)
    {
        if (B->dims[0] != m) return TENSOR_ERR_SHAPE_MISMATCH;
    }
    else if (b_ndim == 2)
    {
        if (B->dims[0] != m) return TENSOR_ERR_SHAPE_MISMATCH;
        k = B->dims[1];
    }
    else
    {
        return TENSOR_ERR_SHAPE_MISMATCH;
    }

    // 检查 X 的形状
    if (b_ndim == 1)
    {
        if (X->ndim != 1 || X->dims[0] != n) return TENSOR_ERR_SHAPE_MISMATCH;
    }
    else
    {
        if (X->ndim != 2 || X->dims[0] != n || X->dims[1] != k) return TENSOR_ERR_SHAPE_MISMATCH;
    }

    // 使用经济 SVD 分解 A: A = U * diag(S) * V^T
    int k_svd = (m < n) ? m : n;  // 奇异值个数
    Tensor *U = tensor_create(2, (int[]){m, k_svd});      // U 形状 (m, k_svd)
    Tensor *S = tensor_create(1, (int[]){k_svd});         // S 形状 (k_svd,)
    Tensor *V = tensor_create(2, (int[]){n, k_svd});      // V 形状 (n, k_svd)
    if (!U || !S || !V)
    {
        tensor_destroy(U);
        tensor_destroy(S);
        tensor_destroy(V);
        return TENSOR_ERR_MEMORY;
    }

    TensorStatus status = tensor_svd(A, U, S, V, 0);  // reduced SVD
    if (status != TENSOR_OK) goto cleanup;

    // 计算容差和伪逆因子
    float max_s = 0.0f;
    for (int i = 0; i < k_svd; ++i)
        if (S->data[i] > max_s) max_s = S->data[i];
    float tol = (max_s > 0) ? (1e-12f * max_s * (m > n ? m : n)) : 0.0f;
    float *S_inv = (float *)malloc(k_svd * sizeof(float));
    if (!S_inv) { status = TENSOR_ERR_MEMORY; goto cleanup; }
    for (int i = 0; i < k_svd; ++i)
        S_inv[i] = (S->data[i] > tol) ? 1.0f / S->data[i] : 0.0f;

    // 计算 U^T * B
    // 先创建转置张量 UT: 形状 (k_svd, m)
    Tensor *UT = tensor_create(2, (int[]){k_svd, m});
    if (!UT) { status = TENSOR_ERR_MEMORY; goto cleanup; }
    status = tensor_permute(U, (int[]){1, 0}, UT);
    if (status != TENSOR_OK) goto cleanup;

    // 创建 UTB 张量，形状取决于 B 的维度
    Tensor *UTB = (b_ndim == 1) ? tensor_create(1, (int[]){k_svd})
                                : tensor_create(2, (int[]){k_svd, k});
    if (!UTB) { status = TENSOR_ERR_MEMORY; goto cleanup; }

    status = tensor_matmul(UT, B, UTB);
    if (status != TENSOR_OK) goto cleanup;

    // 计算 S_inv * UTB -> SUTB
    Tensor *SUTB = (b_ndim == 1) ? tensor_create(1, (int[]){k_svd})
                                 : tensor_create(2, (int[]){k_svd, k});
    if (!SUTB) { status = TENSOR_ERR_MEMORY; goto cleanup; }
    status = tensor_make_unique(SUTB);
    if (status != TENSOR_OK) goto cleanup;

    if (b_ndim == 1)
    {
        for (int i = 0; i < k_svd; ++i)
            SUTB->data[i] = UTB->data[i] * S_inv[i];
    }
    else
    {
        for (int i = 0; i < k_svd; ++i)
        {
            float factor = S_inv[i];
            for (int j = 0; j < k; ++j)
                SUTB->data[i * k + j] = UTB->data[i * k + j] * factor;
        }
    }

    // 计算 X = V * SUTB
    status = tensor_matmul(V, SUTB, X);

cleanup:
    tensor_destroy(U);
    tensor_destroy(S);
    tensor_destroy(V);
    tensor_destroy(UT);
    tensor_destroy(UTB);
    tensor_destroy(SUTB);
    free(S_inv);
    return status;
}
TensorStatus tensor_matrix_rank(const Tensor *src, float tol, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != 2)
        return TENSOR_ERR_INVALID_PARAM;
    int m = src->dims[0];
    int n = src->dims[1];
    int k = (m < n) ? m : n;

    if (out->ndim != 0 && !(out->ndim == 1 && out->dims[0] == 1))
        return TENSOR_ERR_SHAPE_MISMATCH;

    Tensor *S = tensor_create(1, (int[]){k});
    if (!S)
        return TENSOR_ERR_MEMORY;

    // 使用经济 SVD 获取奇异值
    Tensor *U = tensor_create(2, (int[]){m, k});
    Tensor *V = tensor_create(2, (int[]){n, k});
    if (!U || !V)
    {
        tensor_destroy(S);
        tensor_destroy(U);
        tensor_destroy(V);
        return TENSOR_ERR_MEMORY;
    }

    TensorStatus status = tensor_svd(src, U, S, V, 0); // reduced
    if (status != TENSOR_OK)
        goto cleanup;

    float max_s = 0.0f;
    for (int i = 0; i < k; i++)
    {
        if (S->data[i] > max_s)
            max_s = S->data[i];
    }
    float eps = (tol <= 0.0f) ? 1e-12f * max_s * (m > n ? m : n) : tol;

    int rank = 0;
    for (int i = 0; i < k; i++)
        if (S->data[i] > eps)
            rank++;

    status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        goto cleanup;
    out->data[0] = (float)rank;

cleanup:
    tensor_destroy(S);
    tensor_destroy(U);
    tensor_destroy(V);
    return status;
}