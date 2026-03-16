#include "tensor.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ==================== 状态码字符串 ==================== */

const char *tensor_status_to_string(TensorStatus status)
{
    switch (status)
    {
    case TENSOR_OK:
        return "OK";
    case TENSOR_ERR_MEMORY:
        return "Memory allocation failed";
    case TENSOR_ERR_SHAPE_MISMATCH:
        return "Shape mismatch";
    case TENSOR_ERR_INVALID_PARAM:
        return "Invalid parameter";
    case TENSOR_ERR_UNSUPPORTED:
        return "Unsupported operation";
    case TENSOR_ERR_NULL_PTR:
        return "Null pointer";
    case TENSOR_ERR_INDEX_OUT_OF_BOUNDS:
        return "Index out of bounds";
    case TENSOR_ERR_DIV_BY_ZERO:
        return "Division by zero";
    case TENSOR_ERR_NOT_IMPLEMENTED:
        return "Not implemented";
    default:
        return "Unknown error";
    }
}

/* ==================== 写时拷贝辅助 ==================== */

/**
 * 确保张量 t 拥有独占的数据副本（若共享则复制）
 * 返回 TENSOR_OK 或错误码
 */
TensorStatus tensor_make_unique(Tensor *t)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;

    // 外部数据，直接假定可写
    if (!t->ref_count)
        return TENSOR_OK;

    // 已经独占
    if (*(t->ref_count) == 1)
        return TENSOR_OK;

    // 引用计数 > 1，需要复制数据
    float *new_data = (float *)malloc(t->size * sizeof(float));
    if (!new_data)
        return TENSOR_ERR_MEMORY;
    memcpy(new_data, t->data, t->size * sizeof(float));

    // 减少旧数据的引用计数，若归零则释放
    int *old_ref = t->ref_count;
    float *old_data = t->data;
    (*(old_ref))--;
    if (*(old_ref) == 0)
    {
        free(old_data);
        free(old_ref);
    }

    // 设置新数据和新计数器
    t->data = new_data;
    t->ref_count = (int *)malloc(sizeof(int));
    if (!t->ref_count)
    {
        free(new_data);
        return TENSOR_ERR_MEMORY;
    }
    *(t->ref_count) = 1;

    return TENSOR_OK;
}

/* ==================== 基础创建/销毁 ==================== */

Tensor *tensor_create(int ndim, const int *dims)
{
    if (ndim < 0)
        return NULL;
    if (ndim > 0 && !dims)
        return NULL;

    // 标量情况：ndim == 0，此时 dims 可以为 NULL，size = 1
    size_t total_size = (ndim == 0) ? 1 : util_calc_size(dims, ndim);
    if (total_size == 0)
        return NULL; // 不允许空张量（但标量 size=1 允许）

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->data = (float *)calloc(total_size, sizeof(float));
    if (!t->data)
    {
        free(t);
        return NULL;
    }

    if (ndim > 0)
    {
        t->dims = util_copy_ints(dims, ndim);
        if (!t->dims)
        {
            free(t->data);
            free(t);
            return NULL;
        }
        t->strides = util_calc_contiguous_strides(dims, ndim);
        if (!t->strides)
        {
            free(t->dims);
            free(t->data);
            free(t);
            return NULL;
        }
    }
    else
    {
        // 标量
        t->dims = NULL;
        t->strides = NULL;
    }

    t->ndim = ndim;
    t->size = total_size;
    t->ref_count = (int *)malloc(sizeof(int));
    if (!t->ref_count)
    {
        free(t->strides);
        free(t->dims);
        free(t->data);
        free(t);
        return NULL;
    }
    *(t->ref_count) = 1;
    t->owns_dims_strides = 1;

    return t;
}

Tensor *tensor_wrap(float *data, int ndim, const int *dims, const int *strides)
{
    if (!data || ndim < 0)
        return NULL;
    if (ndim > 0 && !dims)
        return NULL;

    size_t total_size = (ndim == 0) ? 1 : util_calc_size(dims, ndim);
    if (total_size == 0)
        return NULL;

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->data = data;
    t->ndim = ndim;
    t->size = total_size;
    t->ref_count = NULL;      // 外部管理
    t->owns_dims_strides = 1; // 我们拥有 dims/strides 副本

    if (ndim > 0)
    {
        t->dims = util_copy_ints(dims, ndim);
        if (!t->dims)
        {
            free(t);
            return NULL;
        }
        if (strides)
        {
            t->strides = util_copy_ints(strides, ndim);
            if (!t->strides)
            {
                free(t->dims);
                free(t);
                return NULL;
            }
        }
        else
        {
            t->strides = NULL; // 标记为连续（由调用者保证实际连续）
        }
    }
    else
    {
        t->dims = NULL;
        t->strides = NULL;
    }

    return t;
}

Tensor *tensor_from_array(const float *data, int ndim, const int *dims)
{
    Tensor *t = tensor_create(ndim, dims);
    if (t)
        memcpy(t->data, data, t->size * sizeof(float));
    return t;
}

void tensor_destroy(Tensor *t)
{
    if (!t)
        return;
    tensor_cleanup(t);
    free(t);
}

void tensor_cleanup(Tensor *t)
{
    if (!t)
        return;
    if (t->ref_count)
    {
        (*(t->ref_count))--;
        if (*(t->ref_count) == 0)
        {
            free(t->data);
            free(t->ref_count);
        }
    }
    if (t->owns_dims_strides)
    {
        free(t->dims);
        free(t->strides);
    }
    // 将指针置空以防止误用
    t->data = NULL;
    t->dims = NULL;
    t->strides = NULL;
    t->ref_count = NULL;
}

Tensor *tensor_clone(const Tensor *src)
{
    if (!src)
        return NULL;
    Tensor *dst = tensor_create(src->ndim, src->dims);
    if (!dst)
        return NULL;
    memcpy(dst->data, src->data, src->size * sizeof(float));
    return dst;
}

Tensor *tensor_view(const Tensor *src, int ndim, const int *dims, const int *strides)
{
    if (!src || ndim < 0)
        return NULL;
    if (ndim > 0 && !dims)
        return NULL;

    // 检查元素总数是否一致
    size_t new_size = (ndim == 0) ? 1 : util_calc_size(dims, ndim);
    if (new_size != src->size)
        return NULL;

    // 分配视图结构
    Tensor *view = (Tensor *)malloc(sizeof(Tensor));
    if (!view)
        return NULL;

    view->data = src->data; // 共享数据
    view->ndim = ndim;
    view->size = new_size;
    view->ref_count = src->ref_count;
    if (view->ref_count)
    {
        (*(view->ref_count))++; // 增加引用计数
    }
    view->owns_dims_strides = 1;

    // 复制 dims
    if (ndim > 0)
    {
        view->dims = util_copy_ints(dims, ndim);
        if (!view->dims)
        {
            if (view->ref_count)
                (*(view->ref_count))--; // 回滚
            free(view);
            return NULL;
        }
    }
    else
    {
        view->dims = NULL;
    }

    // 处理 strides
    if (strides)
    {
        view->strides = util_copy_ints(strides, ndim);
        if (!view->strides)
        {
            free(view->dims);
            if (view->ref_count)
                (*(view->ref_count))--;
            free(view);
            return NULL;
        }
    }
    else
    {
        // 如果调用者未提供步长，且原张量连续，我们可以假设视图也是连续解释（即 reshape）
        // 否则，必须提供步长，这里返回错误
        if (util_is_contiguous(src))
        {
            view->strides = util_calc_contiguous_strides(dims, ndim);
            if (!view->strides && ndim > 0)
            {
                free(view->dims);
                if (view->ref_count)
                    (*(view->ref_count))--;
                free(view);
                return NULL;
            }
        }
        else
        {
            // 原张量不连续且未提供步长，无法确定视图布局
            free(view->dims);
            if (view->ref_count)
                (*(view->ref_count))--;
            free(view);
            return NULL; // 或者可以允许，但风险自负。这里选择严格
        }
    }

    return view;
}

/* ==================== 核心操作 ==================== */

TensorStatus tensor_copy(Tensor *dst, const Tensor *src)
{
    if (!dst || !src)
        return TENSOR_ERR_NULL_PTR;
    if (dst->ndim != src->ndim || !util_shapes_equal(dst->dims, src->dims, dst->ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    // 快速路径：两者连续
    if (util_is_contiguous(src) && util_is_contiguous(dst))
    {
        memcpy(dst->data, src->data, dst->size * sizeof(float));
        return TENSOR_OK;
    }

    // 通用路径：按步长遍历
    int ndim = src->ndim;
    int src_strides[TENSOR_MAX_DIM], dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(dst, dst_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_off = util_offset_from_coords(coords, src_strides, ndim);
        size_t dst_off = util_offset_from_coords(coords, dst_strides, ndim);
        dst->data[dst_off] = src->data[src_off];

        if (util_increment_coords(coords, src->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_contiguous(Tensor *t)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    if (util_is_contiguous(t))
        return TENSOR_OK;

    // 对于外部数据（无引用计数），不能原地修改为连续，因为无法释放原数据
    if (!t->ref_count)
    {
        // 可以尝试分配新数据并替换，但原外部数据会丢失，调用者需自行管理
        // 这里保守返回错误，要求用户自己处理或使用 tensor_clone
        return TENSOR_ERR_UNSUPPORTED;
    }

    // 确保独占数据（如果共享，先复制）
    TensorStatus status = tensor_make_unique(t);
    if (status != TENSOR_OK)
        return status;

    // 分配连续数据
    float *new_data = (float *)malloc(t->size * sizeof(float));
    if (!new_data)
        return TENSOR_ERR_MEMORY;

    // 按行主序复制（使用原步长遍历）
    // 采用索引迭代方式
    int *indices = (int *)calloc(t->ndim, sizeof(int));
    if (!indices)
    {
        free(new_data);
        return TENSOR_ERR_MEMORY;
    }

    size_t linear_idx = 0;
    while (1)
    {
        // 计算原数据偏移
        size_t src_off = 0;
        for (int i = 0; i < t->ndim; ++i)
            src_off += indices[i] * t->strides[i];
        new_data[linear_idx++] = t->data[src_off];

        // 更新索引（行主序递增）
        if (util_increment_coords(indices, t->dims, t->ndim))
            break;
    }
    free(indices);

    // 释放旧数据（通过引用计数）
    // 注意：tensor_make_unique 已确保引用计数为1，所以可以直接释放
    free(t->data);
    // 旧 ref_count 现在指向计数器（值为1），我们也要释放它
    free(t->ref_count);

    // 更新张量
    t->data = new_data;
    t->ref_count = (int *)malloc(sizeof(int));
    if (!t->ref_count)
    {
        free(new_data);
        return TENSOR_ERR_MEMORY;
    }
    *(t->ref_count) = 1;

    // 释放原 strides，设置为 NULL 表示连续
    if (t->owns_dims_strides)
    {
        free(t->strides);
    }
    t->strides = NULL;
    t->owns_dims_strides = 1; // 仍拥有 dims

    return TENSOR_OK;
}

/* ==================== Getter ==================== */

int tensor_ndim(const Tensor *t)
{
    return t ? t->ndim : 0;
}

const int *tensor_dims(const Tensor *t)
{
    return t ? t->dims : NULL;
}

const int *tensor_strides(const Tensor *t)
{
    return t ? t->strides : NULL;
}

size_t tensor_size(const Tensor *t)
{
    return t ? t->size : 0;
}

int tensor_dim_size(const Tensor *t, int axis)
{
    if (!t)
        return -1;
    int ax = util_normalize_axis(axis, t->ndim);
    if (ax < 0)
        return -1;
    return t->dims[ax];
}

size_t tensor_offset(const Tensor *t, const int *indices)
{
    if (!t || !indices)
        return SIZE_MAX;
    size_t offset = 0;
    if (t->strides)
    {
        for (int i = 0; i < t->ndim; ++i)
        {
            int idx = indices[i];
            if (idx < 0)
                idx += t->dims[i];
            if (idx < 0 || idx >= t->dims[i])
                return SIZE_MAX;
            offset += idx * t->strides[i];
        }
    }
    else
    {
        // 连续存储，按行主序计算扁平索引
        size_t mul = 1;
        offset = indices[t->ndim - 1];
        if (offset < 0)
            offset += t->dims[t->ndim - 1];
        if (offset < 0 || offset >= t->dims[t->ndim - 1])
            return SIZE_MAX;
        for (int i = t->ndim - 2; i >= 0; --i)
        {
            mul *= t->dims[i + 1];
            int idx = indices[i];
            if (idx < 0)
                idx += t->dims[i];
            if (idx < 0 || idx >= t->dims[i])
                return SIZE_MAX;
            offset += idx * mul;
        }
    }
    return offset;
}