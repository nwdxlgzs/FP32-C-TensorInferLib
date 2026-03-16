#include "tensor.h"
#include "reduce_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* 根据归约轴和 keepdims 计算输出形状 */
static TensorStatus prepare_output_shape(const Tensor *x, int axis, int keepdims,
                                         int *out_ndim, int *out_dims)
{
    int ndim = x->ndim;
    if (axis == -1)
    {
        *out_ndim = keepdims ? ndim : 0;
        if (keepdims)
        {
            for (int i = 0; i < ndim; i++)
                out_dims[i] = 1;
        }
        return TENSOR_OK;
    }
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (keepdims)
    {
        *out_ndim = ndim;
        memcpy(out_dims, x->dims, ndim * sizeof(int));
        out_dims[ax] = 1;
    }
    else
    {
        *out_ndim = ndim - 1;
        for (int i = 0, j = 0; i < ndim; i++)
        {
            if (i != ax)
                out_dims[j++] = x->dims[i];
        }
    }
    return TENSOR_OK;
}

/* ---------- 通用归约迭代器 ---------- */

/* 内层归约函数类型：对给定基础坐标，计算归约结果并写入 out_val */
typedef void (*reduce_inner_func)(const Tensor *x, int axis, const int *base_coords,
                                  void *user_data, float *out_val);

/* 通用归约主函数 */
static TensorStatus reduce_op_general(const Tensor *x, Tensor *out, int axis, int keepdims,
                                      reduce_inner_func inner_func, void *user_data)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;

    // 计算预期输出形状
    int out_ndim;
    int out_dims[TENSOR_MAX_DIM];
    TensorStatus status = prepare_output_shape(x, axis, keepdims, &out_ndim, out_dims);
    if (status != TENSOR_OK)
        return status;

    // 验证输出形状
    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (!util_shapes_equal(out->dims, out_dims, out_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 确保输出独享数据（写时拷贝）
    status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    int ax = (axis == -1) ? -1 : util_normalize_axis(axis, x->ndim);
    int ndim = x->ndim;

    // 输出坐标数组（行主序递增）
    int out_coords[TENSOR_MAX_DIM] = {0};
    // 输入基础坐标数组（归约轴位置填0）
    int base_coords[TENSOR_MAX_DIM] = {0};

    while (1)
    {
        // 根据 out_coords 构造 base_coords
        if (ax == -1)
        {
            // 归约所有轴：base_coords 全0，无需从 out_coords 复制
            // out_coords 可能为空（标量）或全1（keepdims）
        }
        else if (keepdims)
        {
            memcpy(base_coords, out_coords, ndim * sizeof(int));
        }
        else
        {
            // out_coords 不包含归约轴，需要插入0
            for (int i = 0, j = 0; i < ndim; i++)
            {
                if (i == ax)
                {
                    base_coords[i] = 0;
                }
                else
                {
                    base_coords[i] = out_coords[j++];
                }
            }
        }

        // 计算输出元素的线性偏移（输出是连续的）
        size_t out_offset = 0;
        if (out_ndim > 0)
        {
            size_t mul = 1;
            out_offset = out_coords[out_ndim - 1];
            for (int i = out_ndim - 2; i >= 0; --i)
            {
                mul *= out_dims[i + 1];
                out_offset += out_coords[i] * mul;
            }
        }

        // 调用内层归约函数，结果写入 out->data[out_offset]
        inner_func(x, ax, base_coords, user_data, out->data + out_offset);

        // 递增输出坐标（行主序）
        if (util_increment_coords(out_coords, out_dims, out_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ---------- 各个归约操作的内层函数 ---------- */

static void sum_inner(const Tensor *x, int axis, const int *base_coords,
                      void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        double acc = 0.0;
        for (size_t i = 0; i < x->size; i++)
            acc += x->data[i];
        *out_val = (float)acc;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    double acc = 0.0;
    for (int i = 0; i < inner_size; i++)
    {
        acc += x->data[base_off + i * inner_stride];
    }
    *out_val = (float)acc;
}

static void mean_inner(const Tensor *x, int axis, const int *base_coords,
                       void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        size_t n = x->size;
        if (n == 0)
        {
            *out_val = 0.0f;
            return;
        }
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
            sum += x->data[i];
        *out_val = (float)(sum / n);
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    if (inner_size == 0)
    {
        *out_val = 0.0f;
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < inner_size; i++)
    {
        sum += x->data[base_off + i * inner_stride];
    }
    *out_val = (float)(sum / inner_size);
}

static void prod_inner(const Tensor *x, int axis, const int *base_coords,
                       void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        double acc = 1.0;
        for (size_t i = 0; i < x->size; i++)
            acc *= x->data[i];
        *out_val = (float)acc;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    double acc = 1.0;
    for (int i = 0; i < inner_size; i++)
    {
        acc *= x->data[base_off + i * inner_stride];
    }
    *out_val = (float)acc;
}

static void max_inner(const Tensor *x, int axis, const int *base_coords,
                      void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        float max_val = -INFINITY;
        for (size_t i = 0; i < x->size; i++)
        {
            if (x->data[i] > max_val)
                max_val = x->data[i];
        }
        *out_val = max_val;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    float max_val = -INFINITY;
    for (int i = 0; i < inner_size; i++)
    {
        float v = x->data[base_off + i * inner_stride];
        if (v > max_val)
            max_val = v;
    }
    *out_val = max_val;
}

static void min_inner(const Tensor *x, int axis, const int *base_coords,
                      void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        float min_val = INFINITY;
        for (size_t i = 0; i < x->size; i++)
        {
            if (x->data[i] < min_val)
                min_val = x->data[i];
        }
        *out_val = min_val;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    float min_val = INFINITY;
    for (int i = 0; i < inner_size; i++)
    {
        float v = x->data[base_off + i * inner_stride];
        if (v < min_val)
            min_val = v;
    }
    *out_val = min_val;
}

static void argmax_inner(const Tensor *x, int axis, const int *base_coords,
                         void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        float max_val = -INFINITY;
        size_t max_idx = 0;
        for (size_t i = 0; i < x->size; i++)
        {
            if (x->data[i] > max_val)
            {
                max_val = x->data[i];
                max_idx = i;
            }
        }
        *out_val = (float)max_idx;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    float max_val = -INFINITY;
    int max_idx = 0;
    for (int i = 0; i < inner_size; i++)
    {
        float v = x->data[base_off + i * inner_stride];
        if (v > max_val)
        {
            max_val = v;
            max_idx = i;
        }
    }
    *out_val = (float)max_idx;
}

static void argmin_inner(const Tensor *x, int axis, const int *base_coords,
                         void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        float min_val = INFINITY;
        size_t min_idx = 0;
        for (size_t i = 0; i < x->size; i++)
        {
            if (x->data[i] < min_val)
            {
                min_val = x->data[i];
                min_idx = i;
            }
        }
        *out_val = (float)min_idx;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    float min_val = INFINITY;
    int min_idx = 0;
    for (int i = 0; i < inner_size; i++)
    {
        float v = x->data[base_off + i * inner_stride];
        if (v < min_val)
        {
            min_val = v;
            min_idx = i;
        }
    }
    *out_val = (float)min_idx;
}

static void any_inner(const Tensor *x, int axis, const int *base_coords,
                      void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        for (size_t i = 0; i < x->size; i++)
        {
            if (x->data[i] != 0.0f)
            {
                *out_val = 1.0f;
                return;
            }
        }
        *out_val = 0.0f;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    for (int i = 0; i < inner_size; i++)
    {
        if (x->data[base_off + i * inner_stride] != 0.0f)
        {
            *out_val = 1.0f;
            return;
        }
    }
    *out_val = 0.0f;
}

static void all_inner(const Tensor *x, int axis, const int *base_coords,
                      void *user_data, float *out_val)
{
    (void)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        for (size_t i = 0; i < x->size; i++)
        {
            if (x->data[i] == 0.0f)
            {
                *out_val = 0.0f;
                return;
            }
        }
        *out_val = 1.0f;
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    for (int i = 0; i < inner_size; i++)
    {
        if (x->data[base_off + i * inner_stride] == 0.0f)
        {
            *out_val = 0.0f;
            return;
        }
    }
    *out_val = 1.0f;
}

static void var_inner(const Tensor *x, int axis, const int *base_coords,
                      void *user_data, float *out_val)
{
    int unbiased = *(int *)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        size_t n = x->size;
        if (n == 0)
        {
            *out_val = 0.0f;
            return;
        }
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
            sum += x->data[i];
        double mean = sum / n;
        double sq_sum = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            double d = x->data[i] - mean;
            sq_sum += d * d;
        }
        double denom = unbiased ? (n - 1) : n;
        *out_val = (float)(sq_sum / denom);
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    if (inner_size == 0)
    {
        *out_val = 0.0f;
        return;
    }

    double sum = 0.0;
    for (int i = 0; i < inner_size; i++)
    {
        sum += x->data[base_off + i * inner_stride];
    }
    double mean = sum / inner_size;

    double sq_sum = 0.0;
    for (int i = 0; i < inner_size; i++)
    {
        double d = x->data[base_off + i * inner_stride] - mean;
        sq_sum += d * d;
    }
    double denom = unbiased ? (inner_size - 1) : inner_size;
    *out_val = (float)(sq_sum / denom);
}

static void std_inner(const Tensor *x, int axis, const int *base_coords,
                      void *user_data, float *out_val)
{
    // 复用方差，最后开方
    var_inner(x, axis, base_coords, user_data, out_val);
    *out_val = sqrtf(*out_val);
}

static void norm_inner(const Tensor *x, int axis, const int *base_coords,
                       void *user_data, float *out_val)
{
    float p = *(float *)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    if (axis == -1)
    {
        size_t n = x->size;
        if (p == 0.0f)
        {
            int cnt = 0;
            for (size_t i = 0; i < n; i++)
                if (x->data[i] != 0.0f)
                    cnt++;
            *out_val = (float)cnt;
        }
        else
        {
            double acc = 0.0;
            for (size_t i = 0; i < n; i++)
            {
                acc += powf(fabsf(x->data[i]), p);
            }
            *out_val = (float)pow(acc, 1.0 / p);
        }
        return;
    }

    int inner_size = x->dims[axis];
    int inner_stride = x_strides[axis];
    size_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
    if (p == 0.0f)
    {
        int cnt = 0;
        for (int i = 0; i < inner_size; i++)
        {
            if (x->data[base_off + i * inner_stride] != 0.0f)
                cnt++;
        }
        *out_val = (float)cnt;
    }
    else
    {
        double acc = 0.0;
        for (int i = 0; i < inner_size; i++)
        {
            acc += powf(fabsf(x->data[base_off + i * inner_stride]), p);
        }
        *out_val = (float)pow(acc, 1.0 / p);
    }
}

/* ---------- 对外 API 实现 ---------- */

TensorStatus tensor_sum(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, sum_inner, NULL);
}

TensorStatus tensor_mean(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, mean_inner, NULL);
}

TensorStatus tensor_prod(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, prod_inner, NULL);
}

TensorStatus tensor_max(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, max_inner, NULL);
}

TensorStatus tensor_min(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, min_inner, NULL);
}

TensorStatus tensor_argmax(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, argmax_inner, NULL);
}

TensorStatus tensor_argmin(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, argmin_inner, NULL);
}

TensorStatus tensor_any(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, any_inner, NULL);
}

TensorStatus tensor_all(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    return reduce_op_general(x, out, axis, keepdims, all_inner, NULL);
}

TensorStatus tensor_var(const Tensor *x, Tensor *out, int axis, int keepdims, int unbiased)
{
    return reduce_op_general(x, out, axis, keepdims, var_inner, &unbiased);
}

TensorStatus tensor_std(const Tensor *x, Tensor *out, int axis, int keepdims, int unbiased)
{
    return reduce_op_general(x, out, axis, keepdims, std_inner, &unbiased);
}

TensorStatus tensor_norm(const Tensor *x, Tensor *out, int axis, int keepdims, float p)
{
    return reduce_op_general(x, out, axis, keepdims, norm_inner, &p);
}