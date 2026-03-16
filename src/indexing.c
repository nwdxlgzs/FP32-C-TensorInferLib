#include "tensor.h"
#include "indexing.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ---------- 单个元素访问 ---------- */

TensorStatus tensor_get_item(const Tensor *src, const int *indices, float *out_value)
{
    if (!src || !indices || !out_value)
        return TENSOR_ERR_NULL_PTR;

    size_t off = tensor_offset(src, indices);
    if (off == SIZE_MAX)
        return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;

    *out_value = src->data[off];
    return TENSOR_OK;
}

TensorStatus tensor_set_item(Tensor *dst, const int *indices, float value)
{
    if (!dst || !indices)
        return TENSOR_ERR_NULL_PTR;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    size_t off = tensor_offset(dst, indices);
    if (off == SIZE_MAX)
        return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;

    dst->data[off] = value;
    return TENSOR_OK;
}

/* ---------- 切片（返回视图） ---------- */

TensorStatus tensor_get_slice(const Tensor *src, const int *starts, const int *ends,
                              const int *steps, Tensor *out)
{
    if (!src || !starts || !ends || !out)
        return TENSOR_ERR_NULL_PTR;
    int ndim = src->ndim;

    // 复制步长数组，若 steps 为 NULL 则默认步长为 1
    int *step_arr = (int *)malloc(ndim * sizeof(int));
    if (!step_arr)
        return TENSOR_ERR_MEMORY;
    for (int i = 0; i < ndim; i++)
        step_arr[i] = (steps) ? steps[i] : 1;

    // 获取源步长
    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    int *new_dims = (int *)malloc(ndim * sizeof(int));
    int *new_strides = (int *)malloc(ndim * sizeof(int));
    if (!new_dims || !new_strides)
    {
        free(step_arr);
        free(new_dims);
        free(new_strides);
        return TENSOR_ERR_MEMORY;
    }

    ptrdiff_t data_offset = 0; // 使用有符号类型，确保指针运算安全

    for (int i = 0; i < ndim; i++)
    {
        int dim = src->dims[i];
        int start = starts[i];
        int end = ends[i];
        int step = step_arr[i];

        if (step == 0)
        {
            free(step_arr);
            free(new_dims);
            free(new_strides);
            return TENSOR_ERR_INVALID_PARAM;
        }

        // 负索引处理
        if (start < 0)
            start += dim;
        if (end < 0)
            end += dim;

        // 边界裁剪
        if (start < 0)
            start = 0;
        if (start > dim)
            start = dim;
        if (end < 0)
            end = 0;
        if (end > dim)
            end = dim;

        // 计算切片后该维度的大小
        int size;
        if (step > 0)
        {
            if (start >= end)
                size = 0;
            else
                size = (end - start + step - 1) / step;
        }
        else // step < 0
        {
            if (start <= end)
                size = 0;
            else
                size = (start - end - step - 1) / (-step);
        }

        if (size == 0)
        {
            free(step_arr);
            free(new_dims);
            free(new_strides);
            return TENSOR_ERR_INVALID_PARAM; // 不允许空切片
        }

        new_dims[i] = size;
        new_strides[i] = src_strides[i] * step;
        data_offset += (ptrdiff_t)start * src_strides[i];
    }

    // 计算总元素数
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++)
        total_size *= new_dims[i];

    // 填充输出视图
    out->data = src->data + data_offset;
    out->ndim = ndim;
    out->dims = new_dims;
    out->strides = new_strides;
    out->size = total_size;
    out->ref_count = src->ref_count;
    if (out->ref_count)
        (*(out->ref_count))++;
    out->owns_dims_strides = 1;

    free(step_arr);
    return TENSOR_OK;
}

/* ---------- 高级索引（整数数组） ---------- */

/* 计算多个张量的广播形状，假设所有张量都是整数索引 */
static int broadcast_indices_shape(const Tensor **indices, int num_indices,
                                   int *out_dims, int *out_ndim)
{
    if (num_indices == 0)
    {
        *out_ndim = 0;
        return 1;
    }

    // 以第一个索引张量的形状为基准
    const Tensor *first = indices[0];
    *out_ndim = first->ndim;
    for (int i = 0; i < first->ndim; i++)
        out_dims[i] = first->dims[i];

    for (int k = 1; k < num_indices; k++)
    {
        const Tensor *t = indices[k];
        int temp_ndim;
        int temp_dims[TENSOR_MAX_DIM];
        if (!util_broadcast_shape(out_dims, *out_ndim, t->dims, t->ndim,
                                  temp_dims, &temp_ndim))
            return 0;
        *out_ndim = temp_ndim;
        memcpy(out_dims, temp_dims, temp_ndim * sizeof(int));
    }
    return 1;
}

TensorStatus tensor_advanced_index(const Tensor *src, const Tensor **indices, int num_indices,
                                   Tensor *out)
{
    if (!src || !indices || num_indices <= 0 || !out)
        return TENSOR_ERR_NULL_PTR;
    if (num_indices > src->ndim)
        return TENSOR_ERR_INVALID_PARAM;

    // 计算索引张量的广播形状
    int idx_ndim;
    int idx_dims[TENSOR_MAX_DIM];
    if (!broadcast_indices_shape(indices, num_indices, idx_dims, &idx_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 剩余维度
    int rem_ndim = src->ndim - num_indices;
    int *rem_dims = src->dims + num_indices; // 剩余维度的起始指针

    // 输出形状 = idx_dims + rem_dims
    int out_ndim = idx_ndim + rem_ndim;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < idx_ndim; i++)
        out_dims[i] = idx_dims[i];
    for (int i = 0; i < rem_ndim; i++)
        out_dims[idx_ndim + i] = rem_dims[i];

    // 检查输出形状是否匹配
    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; i++)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    // 获取 src 的有效步长
    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    // 为每个索引张量准备 padded 步长（用于广播）
    int *idx_padded_strides[TENSOR_MAX_DIM];
    int idx_eff_strides[TENSOR_MAX_DIM][TENSOR_MAX_DIM];
    for (int k = 0; k < num_indices; k++)
    {
        util_get_effective_strides(indices[k], idx_eff_strides[k]);
    }

    int out_eff_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(out, out_eff_strides);

    // 遍历输出坐标
    int out_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        // 构建 src 坐标
        int src_coords[TENSOR_MAX_DIM];
        // 前 num_indices 个维度由索引张量提供
        for (int d = 0; d < num_indices; d++)
        {
            // 计算当前索引张量的坐标（从 out_coords 中取前 idx_ndim 个，但索引张量可能广播）
            // 索引张量的形状为 idx_dims，out_coords 的前 idx_ndim 对应广播后的坐标
            // 对于每个索引张量，我们需要从 out_coords 中提取对应广播维度的坐标
            // 由于广播规则，索引张量的每个维度要么是1，要么等于 idx_dims 对应维度
            // 我们需要根据索引张量的形状和步长来计算其元素值
            const Tensor *idx_t = indices[d];
            int idx_ndim_t = idx_t->ndim;
            // 计算索引张量的偏移：从 out_coords 的前 idx_ndim 中，取与 idx_t 维度对应的部分
            size_t idx_off = 0;
            int idx_coord[TENSOR_MAX_DIM];
            // 对齐广播维度：索引张量的维度从后向前与 idx_dims 对齐
            int offset = idx_ndim - idx_ndim_t;
            for (int i = 0; i < idx_ndim_t; i++)
            {
                int coord = out_coords[offset + i];
                // 如果 idx_t 该维度大小为1，则坐标应为0（广播）
                if (idx_t->dims[i] == 1)
                    coord = 0;
                idx_coord[i] = coord;
                idx_off += coord * idx_eff_strides[d][i];
            }
            int index_val = (int)idx_t->data[idx_off];
            // 检查索引是否越界
            if (index_val < 0 || index_val >= src->dims[d])
                return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;
            src_coords[d] = index_val;
        }
        // 剩余维度直接从 out_coords 的后 rem_ndim 复制
        for (int d = 0; d < rem_ndim; d++)
            src_coords[num_indices + d] = out_coords[idx_ndim + d];

        // 计算 src 偏移
        size_t src_off = util_offset_from_coords(src_coords, src_strides, src->ndim);
        // 计算 out 偏移
        size_t out_off = util_offset_from_coords(out_coords, out_eff_strides, out_ndim);

        out->data[out_off] = src->data[src_off];

        // 递增 out_coords
        if (util_increment_coords(out_coords, out_dims, out_ndim))
            break;
    }

    return TENSOR_OK;
}

/* ---------- 布尔掩码选择 ---------- */

TensorStatus tensor_masked_select(const Tensor *src, const Tensor *mask, Tensor *out)
{
    if (!src || !mask || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim != mask->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < src->ndim; i++)
        if (src->dims[i] != mask->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    // 确保 out 连续（因为我们将用线性索引写入）
    TensorStatus status = tensor_contiguous(out);
    if (status != TENSOR_OK)
        return status;

    // 第一次遍历：统计有效元素个数
    int ndim = src->ndim;
    int src_strides[TENSOR_MAX_DIM], mask_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(mask, mask_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    size_t count = 0;
    while (1)
    {
        size_t mask_off = util_offset_from_coords(coords, mask_strides, ndim);
        if (mask->data[mask_off] != 0.0f)
            count++;
        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }

    // 检查输出形状
    if (out->ndim != 1 || (int)out->size != (int)count)
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 第二次遍历：填充输出
    memset(coords, 0, ndim * sizeof(int));
    size_t out_idx = 0;
    while (1)
    {
        size_t src_off = util_offset_from_coords(coords, src_strides, ndim);
        size_t mask_off = util_offset_from_coords(coords, mask_strides, ndim);
        if (mask->data[mask_off] != 0.0f)
            out->data[out_idx++] = src->data[src_off];
        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/* ---------- 按索引赋值（支持广播） ---------- */

TensorStatus tensor_index_put(Tensor *dst, const Tensor **indices, int num_indices,
                              const Tensor *values)
{
    if (!dst || !indices || num_indices <= 0 || !values)
        return TENSOR_ERR_NULL_PTR;
    if (num_indices > dst->ndim)
        return TENSOR_ERR_INVALID_PARAM;

    // 确保 dst 独占数据
    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    // 计算索引张量的广播形状
    int idx_ndim;
    int idx_dims[TENSOR_MAX_DIM];
    if (!broadcast_indices_shape(indices, num_indices, idx_dims, &idx_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 剩余维度
    int rem_ndim = dst->ndim - num_indices;
    int *rem_dims = dst->dims + num_indices;

    // 赋值目标形状 = idx_dims + rem_dims
    int target_ndim = idx_ndim + rem_ndim;
    int target_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < idx_ndim; i++)
        target_dims[i] = idx_dims[i];
    for (int i = 0; i < rem_ndim; i++)
        target_dims[idx_ndim + i] = rem_dims[i];

    // 检查 values 是否可以广播到 target_ndim 形状
    int values_ndim;
    int values_dims[TENSOR_MAX_DIM];
    if (!util_broadcast_shape(values->dims, values->ndim, target_dims, target_ndim,
                              values_dims, &values_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;
    // 广播后的形状应与 target_dims 一致
    if (values_ndim != target_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < target_ndim; i++)
        if (values_dims[i] != target_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    // 获取 dst 的有效步长
    int dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);

    // 准备索引张量的有效步长
    int idx_eff_strides[TENSOR_MAX_DIM][TENSOR_MAX_DIM];
    for (int k = 0; k < num_indices; k++)
    {
        util_get_effective_strides(indices[k], idx_eff_strides[k]);
    }

    // 准备 values 的 padded 步长（广播到 target_ndim）
    int values_padded[TENSOR_MAX_DIM];
    util_fill_padded_strides(values, target_ndim, target_dims, values_padded);

    // 遍历 target 坐标
    int target_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        // 构建 dst 坐标
        int dst_coords[TENSOR_MAX_DIM];
        for (int d = 0; d < num_indices; d++)
        {
            const Tensor *idx_t = indices[d];
            int idx_ndim_t = idx_t->ndim;
            size_t idx_off = 0;
            int offset = idx_ndim - idx_ndim_t;
            for (int i = 0; i < idx_ndim_t; i++)
            {
                int coord = target_coords[offset + i];
                if (idx_t->dims[i] == 1)
                    coord = 0;
                idx_off += coord * idx_eff_strides[d][i];
            }
            int index_val = (int)idx_t->data[idx_off];
            if (index_val < 0 || index_val >= dst->dims[d])
                return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;
            dst_coords[d] = index_val;
        }
        for (int d = 0; d < rem_ndim; d++)
            dst_coords[num_indices + d] = target_coords[idx_ndim + d];

        // 计算 dst 偏移
        size_t dst_off = util_offset_from_coords(dst_coords, dst_strides, dst->ndim);

        // 计算 values 偏移
        size_t val_off = 0;
        for (int i = 0; i < target_ndim; i++)
            val_off += target_coords[i] * values_padded[i];

        dst->data[dst_off] = values->data[val_off];

        // 递增 target_coords
        if (util_increment_coords(target_coords, target_dims, target_ndim))
            break;
    }

    return TENSOR_OK;
}

/* ---------- gather ---------- */

TensorStatus tensor_gather(const Tensor *src, int axis, const Tensor *index, Tensor *out)
{
    if (!src || !index || !out)
        return TENSOR_ERR_NULL_PTR;

    int ax = util_normalize_axis(axis, src->ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    // 检查 index 形状：除 ax 外，其他维度必须与 src 一致（或可广播？这里要求严格相等）
    if (src->ndim != index->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < src->ndim; i++)
    {
        if (i == ax)
            continue;
        if (src->dims[i] != index->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    // 输出形状应与 index 相同
    if (out->ndim != index->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < index->ndim; i++)
        if (out->dims[i] != index->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    int idx_strides[TENSOR_MAX_DIM];
    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(index, idx_strides);
    util_get_effective_strides(out, out_strides);

    int ndim = src->ndim;
    int coords[TENSOR_MAX_DIM] = {0};

    while (1)
    {
        // 从 index 中读取索引值
        size_t idx_off = util_offset_from_coords(coords, idx_strides, ndim);
        int gather_idx = (int)index->data[idx_off];
        if (gather_idx < 0 || gather_idx >= src->dims[ax])
            return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;

        // 构造 src 坐标：除了 ax 维用 gather_idx，其他与 coords 相同
        int src_coords[TENSOR_MAX_DIM];
        memcpy(src_coords, coords, ndim * sizeof(int));
        src_coords[ax] = gather_idx;

        size_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
        size_t out_off = util_offset_from_coords(coords, out_strides, ndim);
        out->data[out_off] = src->data[src_off];

        // 递增坐标
        if (util_increment_coords(coords, index->dims, ndim))
            break;
    }

    return TENSOR_OK;
}

/* ---------- scatter ---------- */

TensorStatus tensor_scatter(Tensor *dst, int axis, const Tensor *index, const Tensor *src)
{
    if (!dst || !index || !src)
        return TENSOR_ERR_NULL_PTR;

    int ax = util_normalize_axis(axis, dst->ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    // 检查 index 形状应与 src 相同（简化要求）
    if (src->ndim != index->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < src->ndim; i++)
        if (src->dims[i] != index->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    // 确保 dst 独占数据
    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    int dst_strides[TENSOR_MAX_DIM];
    int idx_strides[TENSOR_MAX_DIM];
    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);
    util_get_effective_strides(index, idx_strides);
    util_get_effective_strides(src, src_strides);

    int ndim = src->ndim;
    int coords[TENSOR_MAX_DIM] = {0};

    while (1)
    {
        // 从 index 中读取目标位置
        size_t idx_off = util_offset_from_coords(coords, idx_strides, ndim);
        int scatter_idx = (int)index->data[idx_off];
        if (scatter_idx < 0 || scatter_idx >= dst->dims[ax])
            return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;

        // 构造 dst 坐标：除了 ax 维用 scatter_idx，其他与 coords 相同
        int dst_coords[TENSOR_MAX_DIM];
        memcpy(dst_coords, coords, ndim * sizeof(int));
        dst_coords[ax] = scatter_idx;

        size_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim);
        size_t src_off = util_offset_from_coords(coords, src_strides, ndim);
        dst->data[dst_off] = src->data[src_off];

        // 递增坐标
        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }

    return TENSOR_OK;
}