#include "tensor.h"
#include "search_ops.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @file search_ops.c
 * @brief 搜索与排序操作的实现：sort, argsort, unique, searchsorted, topk, kthvalue
 */

/* ==================== 辅助比较函数 ==================== */

/**
 * @brief 升序比较两个浮点数，NaN 视为最大值（放在末尾）
 */
static int cmp_float_asc(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;

    if (isnan(fa) && isnan(fb))
        return 0;
    if (isnan(fa))
        return 1; // NaN > 任何有限数
    if (isnan(fb))
        return -1;
    return (fa > fb) - (fa < fb);
}

/**
 * @brief 带索引的结构，用于 argsort
 */
typedef struct
{
    float value;
    int index;
} FloatWithIndex;

/**
 * @brief 升序比较两个 FloatWithIndex，NaN 视为最大值
 */
static int cmp_fwi_asc(const void *a, const void *b)
{
    const FloatWithIndex *fa = (const FloatWithIndex *)a;
    const FloatWithIndex *fb = (const FloatWithIndex *)b;

    if (isnan(fa->value) && isnan(fb->value))
        return 0;
    if (isnan(fa->value))
        return 1;
    if (isnan(fb->value))
        return -1;
    return (fa->value > fb->value) - (fa->value < fb->value);
}
// 辅助结构：带索引的值
typedef struct
{
    float value;
    int index;
} ValIdx;

// 比较函数：按值升序
static int cmp_val_asc(const void *a, const void *b)
{
    float va = ((const ValIdx *)a)->value;
    float vb = ((const ValIdx *)b)->value;
    if (isnan(va) && isnan(vb))
        return 0;
    if (isnan(va))
        return 1;
    if (isnan(vb))
        return -1;
    return (va > vb) - (va < vb);
}

// 比较函数：按值降序
static int cmp_val_desc(const void *a, const void *b)
{
    return -cmp_val_asc(a, b);
}
/* ==================== tensor_sort ==================== */

TensorStatus tensor_sort(const Tensor *src, int axis, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->size == 0)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim = src->ndim;
    int ax = axis;

    if (axis == -1)
    {
        // 展平排序：输出必须是一维且元素数等于 src->size
        if (out->ndim != 1 || out->size != src->size)
            return TENSOR_ERR_SHAPE_MISMATCH;
    }
    else
    {
        ax = util_normalize_axis(axis, ndim);
        if (ax < 0)
            return TENSOR_ERR_INVALID_PARAM;
        // 排序不改变形状
        if (out->ndim != ndim || !util_shapes_equal(out->dims, src->dims, ndim))
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    // 确保输出独占且连续（虽然我们使用步长写入，但连续性能更好）
    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    /* ----- 情况1：展平排序 ----- */
    if (axis == -1)
    {
        float *tmp = (float *)malloc(src->size * sizeof(float));
        if (!tmp)
            return TENSOR_ERR_MEMORY;

        int coords[TENSOR_MAX_DIM] = {0};
        for (size_t i = 0; i < src->size; ++i)
        {
            ptrdiff_t off = util_offset_from_coords(coords, src_strides, ndim);
            tmp[i] = src->data[off];
            if (util_increment_coords(coords, src->dims, ndim))
                break;
        }

        qsort(tmp, src->size, sizeof(float), cmp_float_asc);
        memcpy(out->data, tmp, src->size * sizeof(float));
        free(tmp);
        return TENSOR_OK;
    }

    /* ----- 情况2：指定轴排序 ----- */
    // 外部维度（除排序轴外）
    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_src_strides[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == ax)
            continue;
        outer_dims[idx] = src->dims[i];
        outer_src_strides[idx] = src_strides[i];
        idx++;
    }

    int inner_len = src->dims[ax];
    int inner_src_stride = src_strides[ax];

    // 获取输出步长
    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(out, out_strides);
    int inner_out_stride = out_strides[ax];

    // 分配临时切片数组
    float *slice = (float *)malloc(inner_len * sizeof(float));
    if (!slice)
        return TENSOR_ERR_MEMORY;

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_base = 0;
        for (int i = 0; i < outer_ndim; ++i)
        {
            src_base += outer_coords[i] * outer_src_strides[i];
        }

        // 读取当前切片
        for (int j = 0; j < inner_len; ++j)
        {
            slice[j] = src->data[src_base + j * inner_src_stride];
        }

        qsort(slice, inner_len, sizeof(float), cmp_float_asc);

        // 构建输出基地址：外部坐标 + 排序轴坐标0
        size_t out_base = 0;
        int out_idx = 0;
        for (int i = 0; i < ndim; ++i)
        {
            if (i == ax)
            {
                out_base += 0; // 排序轴坐标为0
            }
            else
            {
                out_base += outer_coords[out_idx++] * out_strides[i];
            }
        }

        // 写入排序后的切片
        for (int j = 0; j < inner_len; ++j)
        {
            out->data[out_base + j * inner_out_stride] = slice[j];
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }

    free(slice);
    return TENSOR_OK;
}

/* ==================== tensor_argsort ==================== */

TensorStatus tensor_argsort(const Tensor *src, int axis, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->size == 0)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != ndim || !util_shapes_equal(out->dims, src->dims, ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_src_strides[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == ax)
            continue;
        outer_dims[idx] = src->dims[i];
        outer_src_strides[idx] = src_strides[i];
        idx++;
    }

    int inner_len = src->dims[ax];
    int inner_src_stride = src_strides[ax];

    // 获取输出步长
    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(out, out_strides);
    int inner_out_stride = out_strides[ax];

    FloatWithIndex *fwi = (FloatWithIndex *)malloc(inner_len * sizeof(FloatWithIndex));
    if (!fwi)
        return TENSOR_ERR_MEMORY;

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_base = 0;
        for (int i = 0; i < outer_ndim; ++i)
        {
            src_base += outer_coords[i] * outer_src_strides[i];
        }

        for (int j = 0; j < inner_len; ++j)
        {
            fwi[j].value = src->data[src_base + j * inner_src_stride];
            fwi[j].index = j;
        }

        qsort(fwi, inner_len, sizeof(FloatWithIndex), cmp_fwi_asc);

        // 构建输出基地址
        size_t out_base = 0;
        int out_idx = 0;
        for (int i = 0; i < ndim; ++i)
        {
            if (i == ax)
            {
                out_base += 0;
            }
            else
            {
                out_base += outer_coords[out_idx++] * out_strides[i];
            }
        }

        // 写入索引
        for (int j = 0; j < inner_len; ++j)
        {
            out->data[out_base + j * inner_out_stride] = (float)fwi[j].index;
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }

    free(fwi);
    return TENSOR_OK;
}

/* ==================== tensor_unique ==================== */

TensorStatus tensor_unique(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->size == 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 1)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    float *tmp = (float *)malloc(src->size * sizeof(float));
    if (!tmp)
        return TENSOR_ERR_MEMORY;

    int coords[TENSOR_MAX_DIM] = {0};
    for (size_t i = 0; i < src->size; ++i)
    {
        ptrdiff_t off = util_offset_from_coords(coords, src_strides, src->ndim);
        tmp[i] = src->data[off];
        if (util_increment_coords(coords, src->dims, src->ndim))
            break;
    }

    qsort(tmp, src->size, sizeof(float), cmp_float_asc);

    // 去重
    float *dst = out->data;
    size_t write_idx = 0;
    float last = tmp[0];
    int last_is_nan = isnan(last);
    dst[write_idx++] = last;

    for (size_t i = 1; i < src->size; ++i)
    {
        float curr = tmp[i];
        int curr_is_nan = isnan(curr);
        if (curr_is_nan && last_is_nan)
        {
            continue;
        }
        if (curr != last && !(curr_is_nan && last_is_nan))
        {
            last = curr;
            last_is_nan = curr_is_nan;
            dst[write_idx++] = curr;
        }
    }

    free(tmp);

    if (write_idx != out->size)
    {
        return TENSOR_ERR_SHAPE_MISMATCH;
    }

    return TENSOR_OK;
}

/* ==================== tensor_searchsorted ==================== */

TensorStatus tensor_searchsorted(const Tensor *sorted, const Tensor *values,
                                 int right, Tensor *out)
{
    if (!sorted || !values || !out)
        return TENSOR_ERR_NULL_PTR;
    if (sorted->ndim != 1)
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (out->ndim != values->ndim || !util_shapes_equal(out->dims, values->dims, values->ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (sorted->size == 0)
    {
        TensorStatus status = tensor_make_unique(out);
        if (status != TENSOR_OK)
            return status;
        status = tensor_contiguous(out);
        if (status != TENSOR_OK)
            return status;
        memset(out->data, 0, out->size * sizeof(float));
        return TENSOR_OK;
    }

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(out);
    if (status != TENSOR_OK)
        return status;

    int n = (int)sorted->size;

    float *sorted_data;
    int sorted_owns = 0;
    if (util_is_contiguous(sorted))
    {
        sorted_data = sorted->data;
    }
    else
    {
        sorted_data = (float *)malloc(n * sizeof(float));
        if (!sorted_data)
            return TENSOR_ERR_MEMORY;
        int strides[1];
        util_get_effective_strides(sorted, strides);
        for (int i = 0; i < n; ++i)
            sorted_data[i] = sorted->data[i * strides[0]];
        sorted_owns = 1;
    }

    int v_ndim = values->ndim;
    int v_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(values, v_strides);
    int v_coords[TENSOR_MAX_DIM] = {0};

    // 获取输出步长
    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(out, out_strides);

    while (1)
    {
        ptrdiff_t v_off = util_offset_from_coords(v_coords, v_strides, v_ndim);
        float val = values->data[v_off];

        int lo = 0, hi = n;
        if (right)
        {
            while (lo < hi)
            {
                int mid = lo + (hi - lo) / 2;
                if (val < sorted_data[mid])
                    hi = mid;
                else
                    lo = mid + 1;
            }
        }
        else
        {
            while (lo < hi)
            {
                int mid = lo + (hi - lo) / 2;
                if (sorted_data[mid] < val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
        }

        // 计算输出偏移
        ptrdiff_t out_off = util_offset_from_coords(v_coords, out_strides, v_ndim);
        out->data[out_off] = (float)lo;

        if (util_increment_coords(v_coords, values->dims, v_ndim))
            break;
    }

    if (sorted_owns)
        free(sorted_data);
    return TENSOR_OK;
}

/* ==================== tensor_topk ==================== */
TensorStatus tensor_topk(const Tensor *src, int k, int axis, int largest, int sorted,
                         Tensor *values, Tensor *indices)
{
    if (!src || !values || !indices)
        return TENSOR_ERR_NULL_PTR;
    if (k <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim = src->ndim;
    int ax = axis;
    if (ax == -1)
    {
        // 展平：输出为一维，长度为 k
        if (k > (int)src->size)
            return TENSOR_ERR_INVALID_PARAM;
        if (values->ndim != 1 || indices->ndim != 1)
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (values->dims[0] != k || indices->dims[0] != k)
            return TENSOR_ERR_SHAPE_MISMATCH;
    }
    else
    {
        ax = util_normalize_axis(axis, ndim);
        if (ax < 0)
            return TENSOR_ERR_INVALID_PARAM;
        if (k > src->dims[ax])
            return TENSOR_ERR_INVALID_PARAM;
        // 输出形状：除 ax 外与 src 相同，ax 维度变为 k
        int out_ndim = ndim;
        int out_dims[TENSOR_MAX_DIM];
        memcpy(out_dims, src->dims, ndim * sizeof(int));
        out_dims[ax] = k;

        if (values->ndim != out_ndim || !util_shapes_equal(values->dims, out_dims, out_ndim))
            return TENSOR_ERR_SHAPE_MISMATCH;
        if (indices->ndim != out_ndim || !util_shapes_equal(indices->dims, out_dims, out_ndim))
            return TENSOR_ERR_SHAPE_MISMATCH;
    }

    // 确保输出独占且连续（简化处理，也可按步长写入）
    TensorStatus status = tensor_make_unique(values);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(values);
    if (status != TENSOR_OK)
        return status;
    status = tensor_make_unique(indices);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(indices);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    if (ax == -1)
    {
        // 展平情况：将整个张量视为一维
        ValIdx *arr = (ValIdx *)malloc(src->size * sizeof(ValIdx));
        if (!arr)
            return TENSOR_ERR_MEMORY;

        // 填充数组
        int coords[TENSOR_MAX_DIM] = {0};
        for (size_t i = 0; i < src->size; ++i)
        {
            ptrdiff_t off = util_offset_from_coords(coords, src_strides, ndim);
            arr[i].value = src->data[off];
            arr[i].index = i; // 线性索引
            if (util_increment_coords(coords, src->dims, ndim))
                break;
        }

        // 排序
        qsort(arr, src->size, sizeof(ValIdx), largest ? cmp_val_desc : cmp_val_asc);

        // 取前 k 个
        for (int i = 0; i < k; ++i)
        {
            values->data[i] = arr[i].value;
            indices->data[i] = (float)arr[i].index;
        }

        free(arr);
        return TENSOR_OK;
    }

    // 指定轴情况
    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_src_strides[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == ax)
            continue;
        outer_dims[idx] = src->dims[i];
        outer_src_strides[idx] = src_strides[i];
        idx++;
    }

    int inner_len = src->dims[ax];
    int inner_src_stride = src_strides[ax];

    // 获取输出步长（由于已连续，可线性访问）
    int out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(values, out_strides);
    int inner_out_stride = out_strides[ax];

    ValIdx *slice = (ValIdx *)malloc(inner_len * sizeof(ValIdx));
    if (!slice)
        return TENSOR_ERR_MEMORY;

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_base = 0;
        for (int i = 0; i < outer_ndim; ++i)
        {
            src_base += outer_coords[i] * outer_src_strides[i];
        }

        // 读取当前切片
        for (int j = 0; j < inner_len; ++j)
        {
            slice[j].value = src->data[src_base + j * inner_src_stride];
            slice[j].index = j;
        }

        // 排序
        qsort(slice, inner_len, sizeof(ValIdx), largest ? cmp_val_desc : cmp_val_asc);

        // 计算输出基地址（外层坐标）
        size_t out_base = 0;
        int out_idx = 0;
        for (int i = 0; i < ndim; ++i)
        {
            if (i == ax)
            {
                out_base += 0; // 排序轴坐标为0
            }
            else
            {
                out_base += outer_coords[out_idx++] * out_strides[i];
            }
        }

        // 写入前 k 个值
        for (int j = 0; j < k; ++j)
        {
            size_t out_off = out_base + j * inner_out_stride;
            values->data[out_off] = slice[j].value;
            indices->data[out_off] = (float)slice[j].index;
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }

    free(slice);
    return TENSOR_OK;
}

/* ==================== tensor_kthvalue ==================== */
TensorStatus tensor_kthvalue(const Tensor *src, int k, int axis, int keepdims,
                             Tensor *values, Tensor *indices)
{
    if (!src || !values || !indices)
        return TENSOR_ERR_NULL_PTR;
    if (k <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim = src->ndim;
    int ax;
    if (axis == -1)
    {
        // 展平情况
        if (ndim == 0)
            return TENSOR_ERR_INVALID_PARAM; // 标量无法展平
        ax = -1;
    }
    else
    {
        ax = util_normalize_axis(axis, ndim);
        if (ax < 0)
            return TENSOR_ERR_INVALID_PARAM;
    }

    // 计算输出形状（类似归约）
    int out_ndim;
    int out_dims[TENSOR_MAX_DIM];
    if (axis == -1)
    {
        // 展平后返回标量或一维（keepdims）
        out_ndim = keepdims ? ndim : 0;
        if (keepdims)
        {
            for (int i = 0; i < ndim; ++i)
                out_dims[i] = 1;
        }
    }
    else
    {
        if (keepdims)
        {
            out_ndim = ndim;
            memcpy(out_dims, src->dims, ndim * sizeof(int));
            out_dims[ax] = 1;
        }
        else
        {
            out_ndim = ndim - 1;
            for (int i = 0, j = 0; i < ndim; ++i)
            {
                if (i != ax)
                    out_dims[j++] = src->dims[i];
            }
        }
    }

    // 检查输出形状
    if (values->ndim != out_ndim || !util_shapes_equal(values->dims, out_dims, out_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (indices->ndim != out_ndim || !util_shapes_equal(indices->dims, out_dims, out_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 确保输出独占且连续
    TensorStatus status = tensor_make_unique(values);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(values);
    if (status != TENSOR_OK)
        return status;
    status = tensor_make_unique(indices);
    if (status != TENSOR_OK)
        return status;
    status = tensor_contiguous(indices);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);

    if (axis == -1)
    {
        // 展平情况
        ValIdx *arr = (ValIdx *)malloc(src->size * sizeof(ValIdx));
        if (!arr)
            return TENSOR_ERR_MEMORY;

        int coords[TENSOR_MAX_DIM] = {0};
        for (size_t i = 0; i < src->size; ++i)
        {
            ptrdiff_t off = util_offset_from_coords(coords, src_strides, ndim);
            arr[i].value = src->data[off];
            arr[i].index = i;
            if (util_increment_coords(coords, src->dims, ndim))
                break;
        }

        qsort(arr, src->size, sizeof(ValIdx), cmp_val_asc); // 升序
        if (k > (int)src->size)
        {
            free(arr);
            return TENSOR_ERR_INVALID_PARAM;
        }
        values->data[0] = arr[k - 1].value;
        indices->data[0] = (float)arr[k - 1].index;

        free(arr);
        return TENSOR_OK;
    }

    // 指定轴
    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_src_strides[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == ax)
            continue;
        outer_dims[idx] = src->dims[i];
        outer_src_strides[idx] = src_strides[i];
        idx++;
    }

    int inner_len = src->dims[ax];
    int inner_src_stride = src_strides[ax];
    if (k > inner_len)
        return TENSOR_ERR_INVALID_PARAM;

    ValIdx *slice = (ValIdx *)malloc(inner_len * sizeof(ValIdx));
    if (!slice)
        return TENSOR_ERR_MEMORY;

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t src_base = 0;
        for (int i = 0; i < outer_ndim; ++i)
        {
            src_base += outer_coords[i] * outer_src_strides[i];
        }

        for (int j = 0; j < inner_len; ++j)
        {
            slice[j].value = src->data[src_base + j * inner_src_stride];
            slice[j].index = j;
        }

        qsort(slice, inner_len, sizeof(ValIdx), cmp_val_asc); // 升序

        // 构造输出坐标
        int out_coords[TENSOR_MAX_DIM];
        if (keepdims)
        {
            for (int i = 0, j = 0; i < ndim; ++i)
            {
                if (i == ax)
                    out_coords[i] = 0;
                else
                    out_coords[i] = outer_coords[j++];
            }
        }
        else
        {
            memcpy(out_coords, outer_coords, outer_ndim * sizeof(int));
        }

        // 计算输出偏移
        size_t out_off = 0;
        if (out_ndim > 0)
        {
            size_t mul = 1;
            out_off = out_coords[out_ndim - 1];
            for (int i = out_ndim - 2; i >= 0; --i)
            {
                mul *= out_dims[i + 1];
                out_off += out_coords[i] * mul;
            }
        }

        values->data[out_off] = slice[k - 1].value;
        indices->data[out_off] = (float)slice[k - 1].index;

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }

    free(slice);
    return TENSOR_OK;
}