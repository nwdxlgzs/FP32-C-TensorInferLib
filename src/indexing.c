#include "tensor.h"
#include "indexing.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @file indexing.c
 * @brief 索引与高级索引操作的实现
 *
 * 包含单元素访问、切片视图、整数数组高级索引、布尔掩码选择、
 * 按索引赋值（index_put）、gather 和 scatter 等操作。
 * 所有函数均遵循写时拷贝原则，在需要修改数据时先调用 tensor_make_unique。
 */

/* ==================== 辅助函数声明 ==================== */

/**
 * @brief 计算多个索引张量的广播形状
 * @param indices 索引张量数组
 * @param num_indices 索引张量个数
 * @param out_dims 输出广播后形状数组（长度至少为最大维度）
 * @param out_ndim 输出广播后维度数
 * @return 1 表示可广播，0 表示不可广播
 */
static int broadcast_indices_shape(const Tensor **indices, int num_indices,
                                   int *out_dims, int *out_ndim);

/* ==================== 单个元素访问 ==================== */

/**
 * @brief 获取单个元素值
 * @param src 源张量
 * @param indices 索引数组，长度等于 src 的维度
 * @param out_value 输出值
 * @return TensorStatus
 */
TensorStatus tensor_get_item(const Tensor *src, const int *indices, float *out_value)
{
    if (!src || !indices || !out_value)
        return TENSOR_ERR_NULL_PTR;

    ptrdiff_t off = tensor_offset(src, indices);
    if (off == SIZE_MAX)
        return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;

    *out_value = src->data[off];
    return TENSOR_OK;
}

/**
 * @brief 设置单个元素值（可能触发写时拷贝）
 * @param dst 目标张量
 * @param indices 索引数组
 * @param value 要设置的值
 * @return TensorStatus
 */
TensorStatus tensor_set_item(Tensor *dst, const int *indices, float value)
{
    if (!dst || !indices)
        return TENSOR_ERR_NULL_PTR;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    ptrdiff_t off = tensor_offset(dst, indices);
    if (off == SIZE_MAX)
        return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;

    dst->data[off] = value;
    return TENSOR_OK;
}

/* ==================== 切片（返回视图） ==================== */

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
                          const int *steps, Tensor *dst)
{
    // 1. 基础指针校验
    if (!src || !starts || !ends || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    // 2. 维度范围校验 (防止栈数组越界)
    if (ndim < 0 || ndim > TENSOR_MAX_DIM)
        return TENSOR_ERR_INVALID_PARAM;

    // 3. 分配临时资源
    // 注意：当 ndim=0 时，malloc(0) 行为依实现而定，可能返回 NULL，需特殊处理
    int *step_arr = NULL;
    int *new_dims = NULL;
    int *new_strides = NULL;

    if (ndim > 0)
    {
        step_arr = (int *)malloc(ndim * sizeof(int));
        new_dims = (int *)malloc(ndim * sizeof(int));
        new_strides = (int *)malloc(ndim * sizeof(int));

        if (!step_arr || !new_dims || !new_strides)
        {
            goto cleanup_error;
        }
    }

    // 4. 初始化步长数组
    for (int i = 0; i < ndim; i++)
    {
        step_arr[i] = (steps) ? steps[i] : 1;
    }

    // 5. 获取源步长
    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    // 6. 计算切片参数
    ptrdiff_t data_offset = 0;
    size_t total_size = 1;

    for (int i = 0; i < ndim; i++)
    {
        int dim = src->dims[i];
        int start = starts[i];
        int end = ends[i];
        int step = step_arr[i];

        // 步长不能为 0
        if (step == 0)
        {
            goto cleanup_error;
        }

        // 负索引处理 (Python 语义)
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
        int size = 0;
        if (step > 0)
        {
            if (start < end)
            {
                size = (end - start + step - 1) / step;
            }
            else
            {
                size = 0; // 空切片
            }
        }
        else
        { // step < 0
            if (start > end)
            {
                size = (start - end - step - 1) / (-step);
            }
            else
            {
                size = 0; // 空切片
            }
        }

        // 原逻辑：不允许空切片
        if (size == 0)
        {
            goto cleanup_error;
        }

        new_dims[i] = size;
        new_strides[i] = src_strides[i] * step;

        // 累加数据偏移 (防止溢出)
        data_offset += (ptrdiff_t)start * src_strides[i];

        // 累加总大小 (防止溢出)
        if (total_size > 0 && size > 0)
        {
            if (total_size > SIZE_MAX / (size_t)size)
            {
                goto cleanup_error; // 整数溢出
            }
            total_size *= (size_t)size;
        }
    }

    // 7. 构建输出张量
    dst->data = src->data + data_offset;
    dst->ndim = ndim;
    dst->dims = new_dims;       // 所有权转移
    dst->strides = new_strides; // 所有权转移
    dst->size = total_size;

    // 8. 引用计数处理
    dst->ref_count = src->ref_count;
    if (dst->ref_count)
    {
        (*dst->ref_count)++;
    }

    // 9. 标记所有权
    dst->owns_dims_strides = 1;

    // 成功路径：释放临时步长数组 (dims/strides 已转移给 dst)
    if (step_arr)
        free(step_arr);
    return TENSOR_OK;

cleanup_error:
    // 错误路径：释放所有已分配资源
    if (step_arr)
        free(step_arr);
    if (new_dims)
        free(new_dims);
    if (new_strides)
        free(new_strides);
    // 不修改 dst 内容，保持未定义状态
    return TENSOR_ERR_INVALID_PARAM; // 或 TENSOR_ERR_MEMORY 视具体错误而定
}

/* ==================== 高级索引（整数数组） ==================== */

/**
 * @brief 计算多个索引张量的广播形状
 * @param indices 索引张量数组
 * @param num_indices 索引张量个数
 * @param out_dims 输出广播后形状数组（长度至少为最大维度）
 * @param out_ndim 输出广播后维度数
 * @return 1 表示可广播，0 表示不可广播
 */
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

/**
 * @brief 整数数组索引（高级索引）
 * @param src 源张量
 * @param indices 索引张量数组，每个张量元素为整数（float 类型），形状需兼容
 * @param num_indices 索引张量个数
 * @param out 输出张量（新数据）
 * @return TensorStatus
 */
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
            const Tensor *idx_t = indices[d];
            int idx_ndim_t = idx_t->ndim;
            // 计算索引张量的偏移：从 out_coords 的前 idx_ndim 中，取与 idx_t 维度对应的部分
            ptrdiff_t idx_off = 0;
            int offset = idx_ndim - idx_ndim_t;
            for (int i = 0; i < idx_ndim_t; i++)
            {
                int coord = out_coords[offset + i];
                // 如果 idx_t 该维度大小为1，则坐标应为0（广播）
                if (idx_t->dims[i] == 1)
                    coord = 0;
                idx_off += coord * idx_eff_strides[d][i];
            }
            float idx_float = idx_t->data[idx_off];
            TensorStatus st;
            int index_val = tensor_float_to_index(idx_float, src->dims[d], &st);
            if (st != TENSOR_OK)
                return st;
            src_coords[d] = index_val;
        }
        // 剩余维度直接从 out_coords 的后 rem_ndim 复制
        for (int d = 0; d < rem_ndim; d++)
            src_coords[num_indices + d] = out_coords[idx_ndim + d];

        // 计算 src 偏移
        ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, src->ndim);
        // 计算 out 偏移
        ptrdiff_t out_off = util_offset_from_coords(out_coords, out_eff_strides, out_ndim);

        out->data[out_off] = src->data[src_off];

        // 递增 out_coords
        if (util_increment_coords(out_coords, out_dims, out_ndim))
            break;
    }

    return TENSOR_OK;
}

/* ==================== 布尔掩码选择 ==================== */

/**
 * @brief 布尔掩码选择
 * @param src 源张量
 * @param mask 布尔掩码张量（0.0/1.0），形状需与 src 相同
 * @param out 输出1维张量（新数据）
 * @return TensorStatus
 */
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
        ptrdiff_t mask_off = util_offset_from_coords(coords, mask_strides, ndim);
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
        ptrdiff_t src_off = util_offset_from_coords(coords, src_strides, ndim);
        ptrdiff_t mask_off = util_offset_from_coords(coords, mask_strides, ndim);
        if (mask->data[mask_off] != 0.0f)
            out->data[out_idx++] = src->data[src_off];
        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/* ==================== 按索引赋值（支持广播） ==================== */

/**
 * @brief 按索引赋值（可能触发写时拷贝）
 * @param dst 目标张量
 * @param indices 索引张量数组，同 advanced_index
 * @param num_indices 索引个数
 * @param values 要赋值的张量，支持广播
 * @return TensorStatus
 */
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
            ptrdiff_t idx_off = 0;
            int offset = idx_ndim - idx_ndim_t;
            for (int i = 0; i < idx_ndim_t; i++)
            {
                int coord = target_coords[offset + i];
                if (idx_t->dims[i] == 1)
                    coord = 0;
                idx_off += coord * idx_eff_strides[d][i];
            }
            float idx_float = idx_t->data[idx_off];
            TensorStatus st;
            int index_val = tensor_float_to_index(idx_float, dst->dims[d], &st);
            if (st != TENSOR_OK)
                return st;
            dst_coords[d] = index_val;
        }
        for (int d = 0; d < rem_ndim; d++)
            dst_coords[num_indices + d] = target_coords[idx_ndim + d];

        // 计算 dst 偏移
        ptrdiff_t dst_off = util_offset_from_coords(dst_coords, dst_strides, dst->ndim);

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

/* ==================== gather ==================== */

/**
 * @brief 沿指定轴收集元素
 * @param src 源张量
 * @param axis 收集轴
 * @param index 索引张量（元素为整数 float），形状除 axis 外与 src 相同
 * @param out 输出张量
 * @return TensorStatus
 */
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
        ptrdiff_t idx_off = util_offset_from_coords(coords, idx_strides, ndim);

        float gather_float = index->data[idx_off];
        TensorStatus st;
        int gather_val = tensor_float_to_index(gather_float, src->dims[ax], &st);
        if (st != TENSOR_OK)
            return st;

        // 构造 src 坐标：除了 ax 维用 gather_idx，其他与 coords 相同
        int src_coords[TENSOR_MAX_DIM];
        memcpy(src_coords, coords, ndim * sizeof(int));
        src_coords[ax] = gather_val;

        ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
        ptrdiff_t out_off = util_offset_from_coords(coords, out_strides, ndim);
        out->data[out_off] = src->data[src_off];

        // 递增坐标
        if (util_increment_coords(coords, index->dims, ndim))
            break;
    }

    return TENSOR_OK;
}

/* ==================== scatter ==================== */

/**
 * @brief 沿指定轴将 src 的值放入 dst 的索引位置
 * @param dst 目标张量（可能触发写时拷贝）
 * @param axis 轴
 * @param index 索引张量
 * @param src 源值张量
 * @return TensorStatus
 */
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
        ptrdiff_t idx_off = util_offset_from_coords(coords, idx_strides, ndim);

        float scatter_float = index->data[idx_off];
        TensorStatus st;
        int scatter_val = tensor_float_to_index(scatter_float, dst->dims[ax], &st);
        if (st != TENSOR_OK)
            return st;

        // 构造 dst 坐标：除了 ax 维用 scatter_idx，其他与 coords 相同
        int dst_coords[TENSOR_MAX_DIM];
        memcpy(dst_coords, coords, ndim * sizeof(int));
        dst_coords[ax] = scatter_val;

        ptrdiff_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim);
        ptrdiff_t src_off = util_offset_from_coords(coords, src_strides, ndim);
        dst->data[dst_off] = src->data[src_off];

        // 递增坐标
        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }

    return TENSOR_OK;
}

TensorStatus tensor_take(const Tensor *src, const Tensor *indices, Tensor *out)
{
    if (!src || !indices || !out)
        return TENSOR_ERR_NULL_PTR;

    // 输出形状必须与 indices 相同
    if (out->ndim != indices->ndim || !util_shapes_equal(out->dims, indices->dims, indices->ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = indices->ndim;
    int idx_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(indices, idx_strides);
    util_get_effective_strides(out, out_strides);

    // 源张量步长（用于将逻辑坐标转换为物理偏移）
    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        ptrdiff_t idx_off = util_offset_from_coords(coords, idx_strides, ndim);
        float idx_float = indices->data[idx_off];

        TensorStatus st;
        int linear_idx = tensor_float_to_index(idx_float, (int)src->size, &st);
        if (st != TENSOR_OK)
            return st;

        // 将线性索引转换为源张量的逻辑坐标
        int src_coords[TENSOR_MAX_DIM];
        util_coords_from_linear((size_t)linear_idx, src->dims, src->ndim, src_coords);

        // 根据坐标和步长计算源张量物理偏移
        ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, src->ndim);

        ptrdiff_t out_off = util_offset_from_coords(coords, out_strides, ndim);
        out->data[out_off] = src->data[src_off];

        if (util_increment_coords(coords, indices->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_put(Tensor *dst, const Tensor *indices, const Tensor *values, int accumulate)
{
    if (!dst || !indices || !values)
        return TENSOR_ERR_NULL_PTR;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    // 检查 values 是否可以广播到 indices 的形状
    int target_ndim;
    int target_dims[TENSOR_MAX_DIM];
    if (!util_broadcast_shape(values->dims, values->ndim, indices->dims, indices->ndim,
                              target_dims, &target_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (target_ndim != indices->ndim || !util_shapes_equal(target_dims, indices->dims, indices->ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    int ndim = indices->ndim;
    int idx_strides[TENSOR_MAX_DIM], val_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(indices, idx_strides);
    util_fill_padded_strides(values, ndim, indices->dims, val_strides);

    int dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        ptrdiff_t idx_off = util_offset_from_coords(coords, idx_strides, ndim);
        float idx_float = indices->data[idx_off];

        TensorStatus st;
        int linear_idx = tensor_float_to_index(idx_float, (int)dst->size, &st);
        if (st != TENSOR_OK)
            return st;

        ptrdiff_t val_off = util_offset_from_coords(coords, val_strides, ndim);
        float val = values->data[val_off];

        // 将线性索引转换为目标张量的逻辑坐标
        int dst_coords[TENSOR_MAX_DIM];
        util_coords_from_linear((size_t)linear_idx, dst->dims, dst->ndim, dst_coords);

        // 根据坐标和步长计算目标张量物理偏移
        ptrdiff_t dst_off = util_offset_from_coords(dst_coords, dst_strides, dst->ndim);

        if (accumulate)
            dst->data[dst_off] += val;
        else
            dst->data[dst_off] = val;

        if (util_increment_coords(coords, indices->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_nonzero(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    // 第一遍：统计非零元素个数
    int coords[TENSOR_MAX_DIM] = {0};
    size_t count = 0;
    while (1)
    {
        ptrdiff_t off = util_offset_from_coords(coords, src_strides, ndim);
        if (src->data[off] != 0.0f)
            count++;
        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }

    if (out->ndim != 2 || out->dims[0] != (int)count || out->dims[1] != ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int out_strides[2];
    util_get_effective_strides(out, out_strides);
    size_t out_idx = 0;

    memset(coords, 0, ndim * sizeof(int));
    while (1)
    {
        ptrdiff_t off = util_offset_from_coords(coords, src_strides, ndim);
        if (src->data[off] != 0.0f)
        {
            for (int d = 0; d < ndim; d++)
            {
                ptrdiff_t out_off = out_idx * out_strides[0] + d * out_strides[1];
                out->data[out_off] = (float)coords[d];
            }
            out_idx++;
        }
        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }

    return TENSOR_OK;
}