#include "tensor.h"
#include "shape_ops.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ---------- 辅助函数 ---------- */

/**
 * 检查通过给定新维度数组是否可以创建视图（元素总数不变且原张量连续或提供了步长）。
 * 若可以，计算新的步长并填充到 new_strides 中。
 * 返回 1 表示成功，0 表示失败。
 */
static int can_reshape_as_view(const Tensor *src, int ndim, const int *dims,
                               int *new_strides, int *out_ndim, int *out_dims)
{
    if (ndim <= 0)
    {
        // 标量视图
        *out_ndim = 0;
        return 1;
    }

    size_t new_size = util_calc_size(dims, ndim);
    if (new_size != src->size)
        return 0;

    // 如果原张量连续，可以直接计算新步长
    if (util_is_contiguous(src))
    {
        int *strides = util_calc_contiguous_strides(dims, ndim);
        if (!strides)
            return 0;
        memcpy(new_strides, strides, ndim * sizeof(int));
        free(strides);
        return 1;
    }

    // 如果不连续，但提供了合适的步长？这里无法自动推断，返回失败
    return 0;
}

/* ---------- reshape ---------- */

TensorStatus tensor_reshape(Tensor *t, int ndim, const int *dims)
{
    if (!t || (ndim > 0 && !dims))
        return TENSOR_ERR_NULL_PTR;

    size_t new_size = (ndim == 0) ? 1 : util_calc_size(dims, ndim);
    if (new_size != t->size)
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (!util_is_contiguous(t))
        return TENSOR_ERR_UNSUPPORTED;

    TensorStatus status = tensor_make_unique(t);
    if (status != TENSOR_OK)
        return status;

    if (t->owns_dims_strides)
    {
        free(t->dims);
        free(t->strides);
    }

    if (ndim > 0)
    {
        t->dims = util_copy_ints(dims, ndim);
        if (!t->dims)
            return TENSOR_ERR_MEMORY;
        t->strides = util_calc_contiguous_strides(dims, ndim);
        if (!t->strides)
        {
            free(t->dims);
            return TENSOR_ERR_MEMORY;
        }
    }
    else
    {
        t->dims = NULL;
        t->strides = NULL;
    }
    t->ndim = ndim;
    t->owns_dims_strides = 1;
    return TENSOR_OK;
}

TensorStatus tensor_reshape_view(const Tensor *src, int ndim, const int *dims, Tensor *dst)
{
    if (!src || !dst || (ndim > 0 && !dims))
        return TENSOR_ERR_NULL_PTR;

    // 标量视图特殊处理
    if (ndim == 0)
    {
        if (src->size != 1)
            return TENSOR_ERR_SHAPE_MISMATCH;
        Tensor *view = tensor_view(src, 0, NULL, NULL);
        if (!view)
            return TENSOR_ERR_MEMORY;
        *dst = *view;
        free(view);
        return TENSOR_OK;
    }

    int new_strides[TENSOR_MAX_DIM];
    int out_ndim = ndim;
    int out_dims[TENSOR_MAX_DIM];
    memcpy(out_dims, dims, ndim * sizeof(int));

    if (!can_reshape_as_view(src, ndim, dims, new_strides, &out_ndim, out_dims))
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 创建视图
    Tensor *view = tensor_view(src, out_ndim, out_dims, new_strides);
    if (!view)
        return TENSOR_ERR_MEMORY;

    // 将视图内容转交给 dst（假设 dst 是未初始化的干净结构）
    *dst = *view;
    free(view);
    return TENSOR_OK;
}

/* ---------- flatten ---------- */

TensorStatus tensor_flatten(const Tensor *src, int start_axis, int end_axis, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int start = util_normalize_axis(start_axis, ndim);
    int end = util_normalize_axis(end_axis, ndim);
    if (start < 0 || end < 0 || start > end)
        return TENSOR_ERR_INVALID_PARAM;

    int flat_dim = 1;
    for (int i = start; i <= end; ++i)
        flat_dim *= src->dims[i];

    int out_ndim = ndim - (end - start);
    int out_dims[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == start)
        {
            out_dims[idx++] = flat_dim;
            i = end;
        }
        else
        {
            out_dims[idx++] = src->dims[i];
        }
    }

    int new_strides[TENSOR_MAX_DIM];
    if (!can_reshape_as_view(src, out_ndim, out_dims, new_strides, &out_ndim, out_dims))
        return TENSOR_ERR_SHAPE_MISMATCH;

    Tensor *view = tensor_view(src, out_ndim, out_dims, new_strides);
    if (!view)
        return TENSOR_ERR_MEMORY;

    *dst = *view;
    free(view);
    return TENSOR_OK;
}

/* ---------- squeeze ---------- */

TensorStatus tensor_squeeze(const Tensor *src, const int *axes, int num_axes, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int *remove = (int *)calloc(ndim, sizeof(int));
    if (!remove)
        return TENSOR_ERR_MEMORY;

    if (axes == NULL)
    {
        for (int i = 0; i < ndim; ++i)
            if (src->dims[i] == 1)
                remove[i] = 1;
    }
    else
    {
        for (int k = 0; k < num_axes; ++k)
        {
            int ax = util_normalize_axis(axes[k], ndim);
            if (ax < 0)
            {
                free(remove);
                return TENSOR_ERR_INVALID_PARAM;
            }
            if (src->dims[ax] != 1)
            {
                free(remove);
                return TENSOR_ERR_SHAPE_MISMATCH;
            }
            remove[ax] = 1;
        }
    }

    int out_ndim = 0;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < ndim; ++i)
        if (!remove[i])
            out_dims[out_ndim++] = src->dims[i];

    free(remove);

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    int new_strides[TENSOR_MAX_DIM];
    int j = 0;
    for (int i = 0; i < ndim; ++i)
        if (src->dims[i] != 1)
            new_strides[j++] = src_strides[i];

    Tensor *view = tensor_view(src, out_ndim, out_dims, new_strides);
    if (!view)
        return TENSOR_ERR_MEMORY;

    *dst = *view;
    free(view);
    return TENSOR_OK;
}

/* ---------- unsqueeze ---------- */

TensorStatus tensor_unsqueeze(const Tensor *src, int axis, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim + 1);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    int out_ndim = ndim + 1;
    int out_dims[TENSOR_MAX_DIM];
    int out_strides[TENSOR_MAX_DIM];

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    int idx = 0;
    for (int i = 0; i < out_ndim; ++i)
    {
        if (i == ax)
        {
            out_dims[i] = 1;
            out_strides[i] = 0;
        }
        else
        {
            out_dims[i] = src->dims[idx];
            out_strides[i] = src_strides[idx];
            idx++;
        }
    }

    Tensor *view = tensor_view(src, out_ndim, out_dims, out_strides);
    if (!view)
        return TENSOR_ERR_MEMORY;

    *dst = *view;
    free(view);
    return TENSOR_OK;
}

/* ---------- concat ---------- */

TensorStatus tensor_concat(const Tensor **inputs, int num_inputs, int axis, Tensor *output)
{
    if (!inputs || num_inputs < 1 || !output)
        return TENSOR_ERR_NULL_PTR;

    int ndim = inputs[0]->ndim;
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    for (int k = 1; k < num_inputs; ++k)
    {
        if (inputs[k]->ndim != ndim)
            return TENSOR_ERR_SHAPE_MISMATCH;
        for (int i = 0; i < ndim; ++i)
            if (i != ax && inputs[k]->dims[i] != inputs[0]->dims[i])
                return TENSOR_ERR_SHAPE_MISMATCH;
    }

    int out_dims[TENSOR_MAX_DIM];
    memcpy(out_dims, inputs[0]->dims, ndim * sizeof(int));
    out_dims[ax] = 0;
    for (int k = 0; k < num_inputs; ++k)
        out_dims[ax] += inputs[k]->dims[ax];

    if (output->ndim != ndim || !util_shapes_equal(output->dims, out_dims, ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(output, out_strides);

    size_t dst_offset_base = 0;
    for (int k = 0; k < num_inputs; ++k)
    {
        const Tensor *src = inputs[k];
        int src_strides[TENSOR_MAX_DIM];
        util_get_effective_strides(src, src_strides);

        int src_coords[TENSOR_MAX_DIM] = {0};
        while (1)
        {
            size_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
            int out_coords[TENSOR_MAX_DIM];
            memcpy(out_coords, src_coords, ndim * sizeof(int));
            out_coords[ax] += dst_offset_base;
            size_t out_off = util_offset_from_coords(out_coords, out_strides, ndim);
            output->data[out_off] = src->data[src_off];

            if (util_increment_coords(src_coords, src->dims, ndim))
                break;
        }
        dst_offset_base += src->dims[ax];
    }
    return TENSOR_OK;
}

/* ---------- stack ---------- */

TensorStatus tensor_stack(const Tensor **inputs, int num_inputs, int axis, Tensor *output)
{
    if (!inputs || num_inputs < 1 || !output)
        return TENSOR_ERR_NULL_PTR;

    int ndim = inputs[0]->ndim;
    int ax = util_normalize_axis(axis, ndim + 1);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    for (int k = 1; k < num_inputs; ++k)
    {
        if (inputs[k]->ndim != ndim || !util_shapes_equal(inputs[k]->dims, inputs[0]->dims, ndim))
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    int out_ndim = ndim + 1;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < ax; ++i)
        out_dims[i] = inputs[0]->dims[i];
    out_dims[ax] = num_inputs;
    for (int i = ax + 1; i < out_ndim; ++i)
        out_dims[i] = inputs[0]->dims[i - 1];

    if (output->ndim != out_ndim || !util_shapes_equal(output->dims, out_dims, out_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(output, out_strides);

    for (int k = 0; k < num_inputs; ++k)
    {
        const Tensor *src = inputs[k];
        int src_strides[TENSOR_MAX_DIM];
        util_get_effective_strides(src, src_strides);

        int src_coords[TENSOR_MAX_DIM] = {0};
        while (1)
        {
            size_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
            int out_coords[TENSOR_MAX_DIM];
            for (int i = 0; i < ax; ++i)
                out_coords[i] = src_coords[i];
            out_coords[ax] = k;
            for (int i = ax + 1; i < out_ndim; ++i)
                out_coords[i] = src_coords[i - 1];
            size_t out_off = util_offset_from_coords(out_coords, out_strides, out_ndim);
            output->data[out_off] = src->data[src_off];

            if (util_increment_coords(src_coords, src->dims, ndim))
                break;
        }
    }
    return TENSOR_OK;
}

/* ---------- split ---------- */

TensorStatus tensor_split(const Tensor *src, int axis, const int *sizes, int num_splits,
                          Tensor **outputs)
{
    if (!src || !sizes || num_splits < 1 || !outputs)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    int src_dim = src->dims[ax];
    int sum = 0;
    for (int i = 0; i < num_splits; ++i)
        sum += sizes[i];
    if (sum != src_dim)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    int offset = 0;
    for (int s = 0; s < num_splits; ++s)
    {
        int out_dims[TENSOR_MAX_DIM];
        memcpy(out_dims, src->dims, ndim * sizeof(int));
        out_dims[ax] = sizes[s];

        int *dims_copy = util_copy_ints(out_dims, ndim);
        int *strides_copy = util_copy_ints(src_strides, ndim);
        if (!dims_copy || !strides_copy)
        {
            free(dims_copy);
            free(strides_copy);
            return TENSOR_ERR_MEMORY;
        }

        Tensor *sub = (Tensor *)malloc(sizeof(Tensor));
        if (!sub)
        {
            free(dims_copy);
            free(strides_copy);
            return TENSOR_ERR_MEMORY;
        }
        sub->data = src->data + offset * src_strides[ax];
        sub->ndim = ndim;
        sub->dims = dims_copy;
        sub->strides = strides_copy;
        sub->size = util_calc_size(out_dims, ndim);
        sub->ref_count = src->ref_count;
        if (sub->ref_count)
            (*sub->ref_count)++;
        sub->owns_dims_strides = 1;

        outputs[s] = sub;
        offset += sizes[s];
    }
    return TENSOR_OK;
}

/* ---------- slice ---------- */

TensorStatus tensor_slice(const Tensor *src, const int *starts, const int *ends,
                          const int *steps, Tensor *dst)
{
    if (!src || !starts || !ends || !dst)
        return TENSOR_ERR_NULL_PTR;
    int ndim = src->ndim;

    int *step_arr = (int *)malloc(ndim * sizeof(int));
    if (!step_arr)
        return TENSOR_ERR_MEMORY;
    for (int i = 0; i < ndim; i++)
        step_arr[i] = (steps) ? steps[i] : 1;

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

    ptrdiff_t data_offset = 0;

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

        if (start < 0)
            start += dim;
        if (end < 0)
            end += dim;

        if (start < 0)
            start = 0;
        if (start > dim)
            start = dim;
        if (end < 0)
            end = 0;
        if (end > dim)
            end = dim;

        int size;
        if (step > 0)
        {
            if (start >= end)
                size = 0;
            else
                size = (end - start + step - 1) / step;
        }
        else
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
            return TENSOR_ERR_INVALID_PARAM;
        }

        new_dims[i] = size;
        new_strides[i] = src_strides[i] * step;
        data_offset += (ptrdiff_t)start * src_strides[i];
    }

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++)
        total_size *= new_dims[i];

    dst->data = src->data + data_offset;
    dst->ndim = ndim;
    dst->dims = new_dims;
    dst->strides = new_strides;
    dst->size = total_size;
    dst->ref_count = src->ref_count;
    if (dst->ref_count)
        (*(dst->ref_count))++;
    dst->owns_dims_strides = 1;

    free(step_arr);
    return TENSOR_OK;
}

/* ---------- repeat ---------- */

TensorStatus tensor_repeat(const Tensor *src, int axis, int repeats, Tensor *dst)
{
    if (!src || !dst || repeats < 1)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    int out_dims[TENSOR_MAX_DIM];
    memcpy(out_dims, src->dims, ndim * sizeof(int));
    out_dims[ax] *= repeats;

    if (dst->ndim != ndim || !util_shapes_equal(dst->dims, out_dims, ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);

    int src_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
        float val = src->data[src_off];

        int dst_coords[TENSOR_MAX_DIM];
        memcpy(dst_coords, src_coords, ndim * sizeof(int));
        for (int r = 0; r < repeats; ++r)
        {
            dst_coords[ax] = src_coords[ax] * repeats + r;
            size_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim);
            dst->data[dst_off] = val;
        }

        if (util_increment_coords(src_coords, src->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/* ---------- tile ---------- */

TensorStatus tensor_tile(const Tensor *src, const int *reps, Tensor *dst)
{
    if (!src || !reps || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim_src = src->ndim;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < ndim_src; ++i)
        out_dims[i] = src->dims[i] * reps[i];

    if (dst->ndim != ndim_src || !util_shapes_equal(dst->dims, out_dims, ndim_src))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);

    int dst_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        int src_coords[TENSOR_MAX_DIM];
        for (int i = 0; i < ndim_src; ++i)
            src_coords[i] = dst_coords[i] % src->dims[i];

        size_t src_off = util_offset_from_coords(src_coords, src_strides, ndim_src);
        size_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim_src);
        dst->data[dst_off] = src->data[src_off];

        if (util_increment_coords(dst_coords, out_dims, ndim_src))
            break;
    }
    return TENSOR_OK;
}

/* ---------- transpose_axes ---------- */

TensorStatus tensor_transpose_axes(const Tensor *src, const int *axes, Tensor *dst)
{
    if (!src || !dst || !axes)
        return TENSOR_ERR_NULL_PTR;
    int ndim = src->ndim;

    int used[TENSOR_MAX_DIM] = {0};
    for (int i = 0; i < ndim; i++)
    {
        int ax = axes[i];
        if (ax < 0 || ax >= ndim || used[ax])
            return TENSOR_ERR_INVALID_PARAM;
        used[ax] = 1;
        if (dst->dims[i] != src->dims[ax])
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int new_strides[TENSOR_MAX_DIM];
    for (int i = 0; i < ndim; i++)
        new_strides[i] = src_strides[axes[i]];

    Tensor *view = tensor_view(src, ndim, dst->dims, new_strides);
    if (!view)
        return TENSOR_ERR_MEMORY;

    *dst = *view;
    free(view);
    return TENSOR_OK;
}

/* ---------- swapaxes ---------- */

TensorStatus tensor_swapaxes(Tensor *t, int axis1, int axis2)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    int ndim = t->ndim;
    int a1 = util_normalize_axis(axis1, ndim);
    int a2 = util_normalize_axis(axis2, ndim);
    if (a1 < 0 || a2 < 0 || a1 == a2)
        return TENSOR_ERR_INVALID_PARAM;

    TensorStatus status = tensor_make_unique(t);
    if (status != TENSOR_OK)
        return status;

    int tmp_dim = t->dims[a1];
    t->dims[a1] = t->dims[a2];
    t->dims[a2] = tmp_dim;

    if (t->strides)
    {
        int tmp_stride = t->strides[a1];
        t->strides[a1] = t->strides[a2];
        t->strides[a2] = tmp_stride;
    }
    else
    {
        int *strides = util_calc_contiguous_strides(t->dims, ndim);
        if (!strides)
            return TENSOR_ERR_MEMORY;
        t->strides = strides;
        t->owns_dims_strides = 1;
    }
    return TENSOR_OK;
}

/* ---------- flip ---------- */

TensorStatus tensor_flip(const Tensor *src, const int *axes, int num_axes, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int *flip_axes = NULL;
    if (axes == NULL)
    {
        flip_axes = (int *)malloc(ndim * sizeof(int));
        if (!flip_axes)
            return TENSOR_ERR_MEMORY;
        for (int i = 0; i < ndim; ++i)
            flip_axes[i] = i;
        num_axes = ndim;
    }
    else
    {
        flip_axes = util_copy_ints(axes, num_axes);
        if (!flip_axes)
            return TENSOR_ERR_MEMORY;
        for (int i = 0; i < num_axes; ++i)
        {
            int ax = util_normalize_axis(flip_axes[i], ndim);
            if (ax < 0)
            {
                free(flip_axes);
                return TENSOR_ERR_INVALID_PARAM;
            }
            flip_axes[i] = ax;
        }
    }

    if (dst->ndim != ndim || !util_shapes_equal(dst->dims, src->dims, ndim))
    {
        free(flip_axes);
        return TENSOR_ERR_SHAPE_MISMATCH;
    }

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    int *new_strides = util_copy_ints(src_strides, ndim);
    if (!new_strides)
    {
        free(flip_axes);
        return TENSOR_ERR_MEMORY;
    }

    ptrdiff_t data_offset = 0;
    for (int i = 0; i < num_axes; ++i)
    {
        int ax = flip_axes[i];
        new_strides[ax] = -new_strides[ax];
        data_offset += (src->dims[ax] - 1) * src_strides[ax];
    }

    float *new_data = src->data + data_offset;

    Tensor *view = (Tensor *)malloc(sizeof(Tensor));
    if (!view)
    {
        free(flip_axes);
        free(new_strides);
        return TENSOR_ERR_MEMORY;
    }
    view->data = new_data;
    view->ndim = ndim;
    view->dims = util_copy_ints(src->dims, ndim);
    if (!view->dims)
    {
        free(flip_axes);
        free(new_strides);
        free(view);
        return TENSOR_ERR_MEMORY;
    }
    view->strides = new_strides;
    view->size = src->size;
    view->ref_count = src->ref_count;
    if (view->ref_count)
        (*view->ref_count)++;
    view->owns_dims_strides = 1;

    free(flip_axes);
    *dst = *view;
    free(view);
    return TENSOR_OK;
}

/* ---------- pad ---------- */

TensorStatus tensor_pad(const Tensor *src, const int *pad_widths, PadMode mode,
                        float constant_value, Tensor *dst)
{
    if (!src || !pad_widths || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int *pads = util_copy_ints(pad_widths, 2 * ndim);
    if (!pads)
        return TENSOR_ERR_MEMORY;

    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < ndim; ++i)
    {
        int before = pads[2 * i];
        int after = pads[2 * i + 1];
        if (before < 0 || after < 0)
        {
            free(pads);
            return TENSOR_ERR_INVALID_PARAM;
        }
        out_dims[i] = src->dims[i] + before + after;
    }

    if (dst->ndim != ndim || !util_shapes_equal(dst->dims, out_dims, ndim))
    {
        free(pads);
        return TENSOR_ERR_SHAPE_MISMATCH;
    }

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
    {
        free(pads);
        return status;
    }

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);

    int dst_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        int in_range = 1;
        int src_coords[TENSOR_MAX_DIM];
        for (int i = 0; i < ndim; ++i)
        {
            int before = pads[2 * i];
            int d = dst_coords[i] - before;
            if (d < 0 || d >= src->dims[i])
            {
                in_range = 0;
                break;
            }
            src_coords[i] = d;
        }

        size_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim);
        if (in_range)
        {
            size_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
            dst->data[dst_off] = src->data[src_off];
        }
        else
        {
            // 根据填充模式计算源坐标
            int mapped_coords[TENSOR_MAX_DIM];
            int valid = 1;
            for (int i = 0; i < ndim; ++i)
            {
                int before = pads[2 * i];
                int dim = src->dims[i];
                int x = dst_coords[i] - before; // 在原空间中的坐标（可能越界）

                switch (mode)
                {
                case PAD_CONSTANT:
                    // 常量模式已在上层处理，这里不会进入
                    break;
                case PAD_REPLICATE:
                    if (x < 0)
                        x = 0;
                    else if (x >= dim)
                        x = dim - 1;
                    mapped_coords[i] = x;
                    break;
                case PAD_CIRCULAR:
                    if (dim == 0)
                    {
                        valid = 0;
                        break;
                    }
                    x %= dim;
                    if (x < 0)
                        x += dim;
                    mapped_coords[i] = x;
                    break;
                case PAD_REFLECT:
                    if (dim == 1)
                    {
                        // 长度为1时，反射只能取0
                        mapped_coords[i] = 0;
                    }
                    else
                    {
                        int period = 2 * dim - 2;
                        x %= period;
                        if (x < 0)
                            x += period;
                        mapped_coords[i] = (x < dim) ? x : period - x;
                    }
                    break;
                default:
                    valid = 0;
                    break;
                }
                if (!valid)
                    break;
            }

            if (!valid)
            {
                free(pads);
                return TENSOR_ERR_INVALID_PARAM;
            }

            if (mode == PAD_CONSTANT)
            {
                dst->data[dst_off] = constant_value;
            }
            else
            {
                size_t src_off = util_offset_from_coords(mapped_coords, src_strides, ndim);
                dst->data[dst_off] = src->data[src_off];
            }
        }

        if (util_increment_coords(dst_coords, out_dims, ndim))
            break;
    }

    free(pads);
    return TENSOR_OK;
}

/* ---------- cumsum ---------- */

TensorStatus tensor_cumsum(const Tensor *src, int axis, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (dst->ndim != ndim || !util_shapes_equal(dst->dims, src->dims, ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);

    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_src_strides[TENSOR_MAX_DIM];
    int outer_dst_strides[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == ax)
            continue;
        outer_dims[idx] = src->dims[i];
        outer_src_strides[idx] = src_strides[i];
        outer_dst_strides[idx] = dst_strides[i];
        idx++;
    }

    int inner_len = src->dims[ax];
    int inner_src_stride = src_strides[ax];
    int inner_dst_stride = dst_strides[ax];

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_base = 0, dst_base = 0;
        for (int i = 0; i < outer_ndim; ++i)
        {
            src_base += outer_coords[i] * outer_src_strides[i];
            dst_base += outer_coords[i] * outer_dst_strides[i];
        }

        double running = 0.0;
        for (int j = 0; j < inner_len; ++j)
        {
            size_t src_off = src_base + j * inner_src_stride;
            running += src->data[src_off];
            size_t dst_off = dst_base + j * inner_dst_stride;
            dst->data[dst_off] = (float)running;
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ---------- cumprod ---------- */

TensorStatus tensor_cumprod(const Tensor *src, int axis, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (dst->ndim != ndim || !util_shapes_equal(dst->dims, src->dims, ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(dst, dst_strides);

    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_src_strides[TENSOR_MAX_DIM];
    int outer_dst_strides[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == ax)
            continue;
        outer_dims[idx] = src->dims[i];
        outer_src_strides[idx] = src_strides[i];
        outer_dst_strides[idx] = dst_strides[i];
        idx++;
    }

    int inner_len = src->dims[ax];
    int inner_src_stride = src_strides[ax];
    int inner_dst_stride = dst_strides[ax];

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_base = 0, dst_base = 0;
        for (int i = 0; i < outer_ndim; ++i)
        {
            src_base += outer_coords[i] * outer_src_strides[i];
            dst_base += outer_coords[i] * outer_dst_strides[i];
        }

        double running = 1.0;
        for (int j = 0; j < inner_len; ++j)
        {
            size_t src_off = src_base + j * inner_src_stride;
            running *= src->data[src_off];
            size_t dst_off = dst_base + j * inner_dst_stride;
            dst->data[dst_off] = (float)running;
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }
    return TENSOR_OK;
}