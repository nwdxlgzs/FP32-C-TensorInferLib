#include "tensor.h"
#include "shape_ops.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/**
 * @file shape_ops.c
 * @brief 形状操作的实现：reshape, flatten, squeeze, unsqueeze, concat, stack, split, repeat, tile, transpose, flip, pad, cumsum, cumprod
 */

/* ==================== 辅助函数 ==================== */
static void roll_1d(float *data, size_t n, int shift)
{
    if (n == 0)
        return;
    shift %= (int)n;
    if (shift < 0)
        shift += n;
    if (shift == 0)
        return;

    float *tmp = (float *)malloc(shift * sizeof(float));
    if (!tmp)
        return; // 调用者应确保内存足够，这里简化处理
    memcpy(tmp, data + n - shift, shift * sizeof(float));
    memmove(data + shift, data, (n - shift) * sizeof(float));
    memcpy(data, tmp, shift * sizeof(float));
    free(tmp);
}
/**
 * @brief 检查通过给定新维度数组是否可以创建视图（元素总数不变且原张量连续或提供了步长）。
 *        若可以，计算新的步长并填充到 new_strides 中。
 * @return 1 表示成功，0 表示失败。
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

/* ==================== reshape ==================== */

TensorStatus tensor_reshape(Tensor *t, int ndim, const int *dims)
{
    if (!t || (ndim > 0 && !dims))
        return TENSOR_ERR_NULL_PTR;
    if (ndim < 0 || ndim > TENSOR_MAX_DIM)
        return TENSOR_ERR_INVALID_PARAM;
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

TensorStatus tensor_reshape_view(Tensor *dst, const Tensor *src, int ndim, const int *dims)
{
    if (!src || !dst || (ndim > 0 && !dims))
        return TENSOR_ERR_NULL_PTR;
    if (ndim < 0 || ndim > TENSOR_MAX_DIM)
        return TENSOR_ERR_INVALID_PARAM;

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

/* ==================== flatten ==================== */

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
    if (out_ndim > TENSOR_MAX_DIM)
        return TENSOR_ERR_INVALID_PARAM;
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

/* ==================== squeeze ==================== */

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

/* ==================== unsqueeze ==================== */

TensorStatus tensor_unsqueeze(const Tensor *src, int axis, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;
    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim + 1);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;
    int out_ndim = ndim + 1;
    if (out_ndim > TENSOR_MAX_DIM)
        return TENSOR_ERR_INVALID_PARAM;
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

/* ==================== concat ==================== */

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
            ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
            int out_coords[TENSOR_MAX_DIM];
            memcpy(out_coords, src_coords, ndim * sizeof(int));
            out_coords[ax] += dst_offset_base;
            ptrdiff_t out_off = util_offset_from_coords(out_coords, out_strides, ndim);
            output->data[out_off] = src->data[src_off];

            if (util_increment_coords(src_coords, src->dims, ndim))
                break;
        }
        dst_offset_base += src->dims[ax];
    }
    return TENSOR_OK;
}

/* ==================== stack ==================== */

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
        if (inputs[k]->ndim != ndim)
            return TENSOR_ERR_SHAPE_MISMATCH;
        for (int i = 0; i < ndim; ++i)
        {
            if (inputs[k]->dims[i] != inputs[0]->dims[i])
                return TENSOR_ERR_SHAPE_MISMATCH;
        }
    }
    int out_ndim = ndim + 1;
    if (out_ndim > TENSOR_MAX_DIM)
        return TENSOR_ERR_INVALID_PARAM;
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
            ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
            int out_coords[TENSOR_MAX_DIM];
            for (int i = 0; i < ax; ++i)
                out_coords[i] = src_coords[i];
            out_coords[ax] = k;
            for (int i = ax + 1; i < out_ndim; ++i)
                out_coords[i] = src_coords[i - 1];
            ptrdiff_t out_off = util_offset_from_coords(out_coords, out_strides, out_ndim);
            output->data[out_off] = src->data[src_off];

            if (util_increment_coords(src_coords, src->dims, ndim))
                break;
        }
    }
    return TENSOR_OK;
}

/* ==================== split ==================== */

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

/* ==================== repeat ==================== */

TensorStatus tensor_repeat(const Tensor *src, int axis, int repeats, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;
    if (repeats < 1)
        return TENSOR_ERR_INVALID_PARAM;

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
        ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
        float val = src->data[src_off];

        int dst_coords[TENSOR_MAX_DIM];
        memcpy(dst_coords, src_coords, ndim * sizeof(int));
        for (int r = 0; r < repeats; ++r)
        {
            dst_coords[ax] = src_coords[ax] * repeats + r;
            ptrdiff_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim);
            dst->data[dst_off] = val;
        }

        if (util_increment_coords(src_coords, src->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/* ==================== tile ==================== */

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

        ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, ndim_src);
        ptrdiff_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim_src);
        dst->data[dst_off] = src->data[src_off];

        if (util_increment_coords(dst_coords, out_dims, ndim_src))
            break;
    }
    return TENSOR_OK;
}

/* ==================== transpose_axes ==================== */

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

/* ==================== swapaxes ==================== */

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

/* ==================== flip ==================== */

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

/* ==================== pad ==================== */

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

        ptrdiff_t dst_off = util_offset_from_coords(dst_coords, dst_strides, ndim);
        if (in_range)
        {
            ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
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
                ptrdiff_t src_off = util_offset_from_coords(mapped_coords, src_strides, ndim);
                dst->data[dst_off] = src->data[src_off];
            }
        }

        if (util_increment_coords(dst_coords, out_dims, ndim))
            break;
    }

    free(pads);
    return TENSOR_OK;
}

/* ==================== cumsum ==================== */

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

/* ==================== cumprod ==================== */

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
TensorStatus tensor_broadcast_to(const Tensor *src, int ndim, const int *dims, Tensor *out)
{
    if (!src || !out || (ndim > 0 && !dims))
        return TENSOR_ERR_NULL_PTR;
    if (ndim < 0)
        return TENSOR_ERR_INVALID_PARAM;

    int src_ndim = src->ndim;
    // 从右向左检查兼容性
    int i_src = src_ndim - 1;
    int i_tar = ndim - 1;
    while (i_src >= 0 && i_tar >= 0)
    {
        int src_dim = src->dims[i_src];
        int tar_dim = dims[i_tar];
        if (src_dim != tar_dim && src_dim != 1 && tar_dim != 1)
            return TENSOR_ERR_SHAPE_MISMATCH;
        i_src--;
        i_tar--;
    }
    if (i_src >= 0)
        return TENSOR_ERR_SHAPE_MISMATCH;
    while (i_tar >= 0)
    {
        if (dims[i_tar] != 1)
            return TENSOR_ERR_SHAPE_MISMATCH;
        i_tar--;
    }

    // 手动构建视图结构
    Tensor *view = (Tensor *)malloc(sizeof(Tensor));
    if (!view)
        return TENSOR_ERR_MEMORY;

    view->data = src->data;
    view->ndim = ndim;
    view->dims = util_copy_ints(dims, ndim);
    if (!view->dims)
    {
        free(view);
        return TENSOR_ERR_MEMORY;
    }
    view->strides = (int *)malloc(ndim * sizeof(int));
    if (!view->strides)
    {
        free(view->dims);
        free(view);
        return TENSOR_ERR_MEMORY;
    }

    // 计算源张量的连续步长（行主序）
    int src_contiguous_strides[TENSOR_MAX_DIM];
    int stride = 1;
    for (int i = src_ndim - 1; i >= 0; --i)
    {
        src_contiguous_strides[i] = stride;
        stride *= src->dims[i];
    }

    int offset = ndim - src_ndim;
    // 左边新增维度步长为0
    for (int i = 0; i < offset; ++i)
        view->strides[i] = 0;

    // 剩余维度继承自源连续步长，但若源维度为1，则步长设为0
    for (int i = 0; i < src_ndim; ++i)
    {
        int idx = offset + i;
        if (src->dims[i] == 1)
        {
            view->strides[idx] = 0;
        }
        else
        {
            view->strides[idx] = src_contiguous_strides[i];
        }
    }

    view->size = util_calc_size(dims, ndim);
    view->ref_count = src->ref_count;
    if (view->ref_count)
        (*view->ref_count)++;
    view->owns_dims_strides = 1;

    *out = *view;
    free(view);
    return TENSOR_OK;
}

TensorStatus tensor_roll(const Tensor *src, const int *shifts, int num_axes,
                         const int *axes, Tensor *out)
{
    if (!src || !shifts || !out)
        return TENSOR_ERR_NULL_PTR;
    if (num_axes <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    // 检查输出形状必须与 src 相同
    if (out->ndim != src->ndim || !util_shapes_equal(out->dims, src->dims, src->ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = src->ndim;
    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(out, out_strides);

    // ---------- 情况1：对所有轴滚动（展平处理） ----------
    if (axes == NULL)
    {
        // 要求 num_axes == 1，即一个 shift 作用于所有轴
        if (num_axes != 1)
            return TENSOR_ERR_INVALID_PARAM;
        int shift = shifts[0];
        size_t n = src->size;
        if (n == 0)
            return TENSOR_OK;

        // 将 src 按线性顺序复制到临时数组
        float *tmp = (float *)malloc(n * sizeof(float));
        if (!tmp)
            return TENSOR_ERR_MEMORY;

        int coords[TENSOR_MAX_DIM] = {0};
        for (size_t i = 0; i < n; ++i)
        {
            ptrdiff_t off = util_offset_from_coords(coords, src_strides, ndim);
            tmp[i] = src->data[off];
            if (util_increment_coords(coords, src->dims, ndim))
                break;
        }

        // 滚动一维数组
        roll_1d(tmp, n, shift);

        // 写回 out
        memset(coords, 0, ndim * sizeof(int));
        for (size_t i = 0; i < n; ++i)
        {
            ptrdiff_t off = util_offset_from_coords(coords, out_strides, ndim);
            out->data[off] = tmp[i];
            util_increment_coords(coords, out->dims, ndim);
        }
        free(tmp);
        return TENSOR_OK;
    }

    // ---------- 情况2：指定轴滚动 ----------
    // 检查 axes 和 shifts 长度一致
    if (num_axes != 0 && !axes)
        return TENSOR_ERR_INVALID_PARAM; // 实际上前面已经处理了 axes==NULL 的情况，这里不会执行到

    // 归一化轴并收集 shift 映射
    int shift_per_axis[TENSOR_MAX_DIM] = {0};
    int axis_handled[TENSOR_MAX_DIM] = {0};
    for (int i = 0; i < num_axes; ++i)
    {
        int ax = util_normalize_axis(axes[i], ndim);
        if (ax < 0 || axis_handled[ax])
            return TENSOR_ERR_INVALID_PARAM;
        axis_handled[ax] = 1;
        shift_per_axis[ax] = shifts[i];
    }

    // 遍历输出坐标，计算源坐标
    int out_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        // 计算源坐标：out_coords[ax] -> (out_coords[ax] - shift) mod dim
        int src_coords[TENSOR_MAX_DIM];
        for (int i = 0; i < ndim; ++i)
        {
            if (axis_handled[i])
            {
                int dim = src->dims[i];
                int s = shift_per_axis[i];
                int coord = out_coords[i] - s;
                coord %= dim;
                if (coord < 0)
                    coord += dim;
                src_coords[i] = coord;
            }
            else
            {
                src_coords[i] = out_coords[i];
            }
        }

        ptrdiff_t src_off = util_offset_from_coords(src_coords, src_strides, ndim);
        ptrdiff_t out_off = util_offset_from_coords(out_coords, out_strides, ndim);
        out->data[out_off] = src->data[src_off];

        if (util_increment_coords(out_coords, out->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_movedim(const Tensor *src, const int *src_axes, int num_axes,
                            const int *dst_positions, Tensor *out)
{
    if (!src || !src_axes || !dst_positions || !out)
        return TENSOR_ERR_NULL_PTR;
    if (num_axes <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim = src->ndim;
    // 归一化并检查重复和范围
    int used_src[TENSOR_MAX_DIM] = {0};
    int used_dst[TENSOR_MAX_DIM] = {0};
    int norm_src_axes[TENSOR_MAX_DIM];
    int norm_dst_pos[TENSOR_MAX_DIM];
    for (int i = 0; i < num_axes; ++i)
    {
        int ax = util_normalize_axis(src_axes[i], ndim);
        if (ax < 0)
            return TENSOR_ERR_INVALID_PARAM;
        if (used_src[ax])
            return TENSOR_ERR_INVALID_PARAM;
        used_src[ax] = 1;
        norm_src_axes[i] = ax;

        int pos = util_normalize_axis(dst_positions[i], ndim);
        if (pos < 0)
            return TENSOR_ERR_INVALID_PARAM;
        if (used_dst[pos])
            return TENSOR_ERR_INVALID_PARAM;
        used_dst[pos] = 1;
        norm_dst_pos[i] = pos;
    }

    // 构建 perm 数组：perm[new_axis] = old_axis
    int perm[TENSOR_MAX_DIM];
    int remaining_axes[TENSOR_MAX_DIM];
    int rem_count = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (!used_src[i])
            remaining_axes[rem_count++] = i;
    }

    int rem_idx = 0;
    for (int new_ax = 0; new_ax < ndim; ++new_ax)
    {
        int found = -1;
        for (int i = 0; i < num_axes; ++i)
        {
            if (norm_dst_pos[i] == new_ax)
            {
                found = norm_src_axes[i];
                break;
            }
        }
        if (found != -1)
        {
            perm[new_ax] = found;
        }
        else
        {
            perm[new_ax] = remaining_axes[rem_idx++];
        }
    }

    // 计算新 dims 和 strides
    int *new_dims = (int *)malloc(ndim * sizeof(int));
    int *new_strides = (int *)malloc(ndim * sizeof(int));
    if (!new_dims || !new_strides)
    {
        free(new_dims);
        free(new_strides);
        return TENSOR_ERR_MEMORY;
    }

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    for (int i = 0; i < ndim; ++i)
    {
        int old_ax = perm[i];
        new_dims[i] = src->dims[old_ax];
        new_strides[i] = src_strides[old_ax];
    }

    // 创建视图
    Tensor *view = tensor_view(src, ndim, new_dims, new_strides);
    free(new_dims);
    free(new_strides);
    if (!view)
        return TENSOR_ERR_MEMORY;

    *out = *view;
    free(view);
    return TENSOR_OK;
}