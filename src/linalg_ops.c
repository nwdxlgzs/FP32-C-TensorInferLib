#include "tensor.h"
#include "linalg_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/* ---------- 矩阵乘法 (支持批量广播) ---------- */

// 检查矩阵部分是否连续（忽略大小为1的维度）
static int is_contiguous_matrix_part(const Tensor *t, int start_axis)
{
    // 假设 t 的维度 >= start_axis，且最后两维为矩阵维度
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

// 分块矩阵乘法核心（假设 a、b、c 均为行主序连续存储）

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

// ========== 矩阵乘法主函数 ==========
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
        // a: (K,)  b: (..., K, N)  -> out: (..., N)
        int K = (int)a->size;
        int b_K = b->dims[b_ndim - 2];
        if (K != b_K)
            return TENSOR_ERR_SHAPE_MISMATCH;

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

        // 获取步长（略，与原代码相同）
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
        // a: (..., M, K)  b: (K,)  -> out: (..., M)
        int K = (int)b->size;
        int a_K = a->dims[a_ndim - 1];
        if (K != a_K)
            return TENSOR_ERR_SHAPE_MISMATCH;

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

    /* 批量维度广播 */
    int batch_ndim;
    int batch_dims[TENSOR_MAX_DIM];
    if (!util_broadcast_shape(a->dims, a_batch_ndim, b->dims, b_batch_ndim,
                              batch_dims, &batch_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

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
    // 矩阵部分的起始轴索引为 batch_ndim（即最后两维）
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

            // 调用分块乘法，直接使用连续偏移量作为子矩阵指针
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

/* ---------- 批量矩阵乘法（严格三维） ---------- */

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

/* ---------- 向量点积 ---------- */

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

/* ---------- 向量外积 ---------- */

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

/* ---------- 张量缩并（tensordot） ---------- */

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
    /* 获取 a 的有效步长 */
    int a_eff_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(a, a_eff_strides);
    /* 收集 a 的保留轴和缩并轴 */
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
    /* 缩并轴数量应与 naxes 相等 */
    if (red_idx != naxes)
        return TENSOR_ERR_INVALID_PARAM; /* 应该不会发生 */

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

        /* 递增输出坐标 */
        if (util_increment_coords(out_coords, out_dims, out_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ---------- 矩阵转置（复制数据） ---------- */

TensorStatus tensor_transpose(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 2)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim = src->ndim;
    /* 验证输出形状：交换最后两维 */
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
        /* 构造源坐标：交换最后两维 */
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

/* ---------- 一般转置（按给定轴顺序重排，复制数据） ---------- */

TensorStatus tensor_permute(const Tensor *src, const int *axes, Tensor *out)
{
    if (!src || !out || !axes)
        return TENSOR_ERR_NULL_PTR;
    int ndim = src->ndim;
    /* 验证 axes 是 [0, ndim-1] 的一个排列 */
    int used[TENSOR_MAX_DIM] = {0};
    for (int i = 0; i < ndim; i++)
    {
        int ax = axes[i];
        if (ax < 0 || ax >= ndim || used[ax])
            return TENSOR_ERR_INVALID_PARAM;
        used[ax] = 1;
        /* 检查输出形状是否匹配：out->dims[i] 应等于 src->dims[axes[i]] */
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
        /* 构造源坐标：根据 axes 将输出坐标映射回源 */
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

/* ---------- 提取对角线 / 创建对角矩阵 ---------- */

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

        /* 填充 0 */
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

/* ---------- 迹 ---------- */

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

    /* 输出形状：移除 ax1 和 ax2 */
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
        /* 构建源基础坐标 */
        int src_base[TENSOR_MAX_DIM];
        int out_idx = 0;
        for (int i = 0; i < ndim; i++)
        {
            if (i == ax1 || i == ax2)
                src_base[i] = 0; /* 占位，将在内层循环设置 */
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

/* ---------- 矩阵求逆（精度问题选择了double，可以用高斯消元或者LU分解） ---------- */

TensorStatus tensor_inv(const Tensor *src, Tensor *out)
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

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    // 将输入矩阵复制到连续的双精度数组 A（行主序）
    double *A = (double *)malloc(n * n * sizeof(double));
    if (!A)
        return TENSOR_ERR_MEMORY;
    int src_strides[2];
    util_get_effective_strides(src, src_strides);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = src->data[i * src_strides[0] + j * src_strides[1]];

    // 部分主元 LU 分解
    int *piv = (int *)malloc(n * sizeof(int));
    if (!piv)
    {
        free(A);
        return TENSOR_ERR_MEMORY;
    }
    const double eps = 1e-12;
    for (int k = 0; k < n; k++)
    {
        // 寻找主元
        int pivot = k;
        double max_val = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; i++)
        {
            if (fabs(A[i * n + k]) > max_val)
            {
                max_val = fabs(A[i * n + k]);
                pivot = i;
            }
        }
        if (max_val < eps)
        {
            free(A);
            free(piv);
            return TENSOR_ERR_DIV_BY_ZERO; // 矩阵奇异
        }
        piv[k] = pivot;
        if (pivot != k)
        {
            // 交换行
            for (int j = 0; j < n; j++)
            {
                double tmp = A[k * n + j];
                A[k * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
        }

        // 消元
        for (int i = k + 1; i < n; i++)
        {
            double factor = A[i * n + k] / A[k * n + k];
            A[i * n + k] = factor; // 存储乘子到 L 部分
            for (int j = k + 1; j < n; j++)
            {
                A[i * n + j] -= factor * A[k * n + j];
            }
        }
    }

    // 初始化逆矩阵为单位阵（双精度）
    double *inv = (double *)calloc(n * n, sizeof(double));
    if (!inv)
    {
        free(A);
        free(piv);
        return TENSOR_ERR_MEMORY;
    }
    for (int i = 0; i < n; i++)
        inv[i * n + i] = 1.0;

    // 对每一列求解 A * X = I
    double *y = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *bb = (double *)malloc(n * sizeof(double));
    if (!y || !x || !bb)
    {
        free(y);
        free(x);
        free(bb);
        free(inv);
        free(A);
        free(piv);
        return TENSOR_ERR_MEMORY;
    }

    for (int col = 0; col < n; col++)
    {
        // 取右端项（单位阵的第 col 列）
        for (int i = 0; i < n; i++)
            bb[i] = inv[i * n + col];

        // 应用行置换：按分解时的交换顺序正向处理
        for (int k = 0; k < n; k++)
        {
            if (piv[k] != k)
            {
                double tmp = bb[k];
                bb[k] = bb[piv[k]];
                bb[piv[k]] = tmp;
            }
        }

        // 前代解 L * y = bb  (L 单位下三角，存储在 A 的下三角部分)
        y[0] = bb[0];
        for (int i = 1; i < n; i++)
        {
            double sum = bb[i];
            for (int j = 0; j < i; j++)
                sum -= A[i * n + j] * y[j];
            y[i] = sum;
        }

        // 回代解 U * x = y  (U 是上三角，包括对角线)
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = y[i];
            for (int j = i + 1; j < n; j++)
                sum -= A[i * n + j] * x[j];
            x[i] = sum / A[i * n + i];
        }

        // 将解写入 inv 的第 col 列
        for (int i = 0; i < n; i++)
            inv[i * n + col] = x[i];
    }

    free(y);
    free(x);
    free(bb);

    // 将逆矩阵写回输出张量（考虑步长）
    int out_strides[2];
    util_get_effective_strides(out, out_strides);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out->data[i * out_strides[0] + j * out_strides[1]] = (float)inv[i * n + j];

    free(A);
    free(piv);
    free(inv);
    return TENSOR_OK;
}