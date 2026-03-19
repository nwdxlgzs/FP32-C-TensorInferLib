#include "tensor.h"
#include "reduce_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

/**
 * @file reduce_ops.c
 * @brief 归约操作的实现：求和、均值、最值、方差、中位数等。
 *
 * 本文件实现了所有归约相关的操作，利用通用归约迭代器 reduce_op_general 来避免代码重复。
 * 每个具体的归约操作定义了一个内层函数（如 sum_inner），由 reduce_op_general 调用。
 */

/* ==================== 辅助函数 ==================== */

/**
 * @brief 根据归约轴和 keepdims 计算输出形状
 * @param x         输入张量
 * @param axis      归约轴（-1 表示所有轴）
 * @param keepdims  是否保留维度
 * @param out_ndim  输出维度数（输出参数）
 * @param out_dims  输出维度数组（长度至少为 x->ndim）
 * @return TensorStatus  TENSOR_OK 或错误码
 */
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

/**
 * @brief 比较两个浮点数，用于 qsort
 */
static int compare_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/**
 * @brief 根据分位数插值方法计算分位数值
 * @param sorted 已排序的数组
 * @param n      数组长度
 * @param q      分位数 [0,1]
 * @param interp 插值方法
 * @return 分位数值
 */
static float interpolate_quantile(float *sorted, int n, float q, QuantileInterp interp)
{
    if (n == 0)
        return 0.0f;
    if (n == 1)
        return sorted[0];

    float index = q * (n - 1);
    int lo = (int)index;
    int hi = lo + 1;
    float weight = index - lo;

    switch (interp)
    {
    case QUANTILE_LOWER:
        return sorted[lo];
    case QUANTILE_HIGHER:
    {
        int idx = (int)ceilf(index);
        if (idx >= n)
            idx = n - 1;
        return sorted[idx];
    }
    case QUANTILE_MIDPOINT:
        if (weight == 0.0f)
        {
            return sorted[lo];
        }
        else
        {
            return (sorted[lo] + sorted[hi]) * 0.5f;
        }
    case QUANTILE_NEAREST:
        return (weight < 0.5f) ? sorted[lo] : sorted[hi];
    case QUANTILE_LINEAR:
    default:
        return sorted[lo] * (1 - weight) + sorted[hi] * weight;
    }
}

/* ==================== 通用归约迭代器 ==================== */

/**
 * @brief 内层归约函数类型
 * @param x            输入张量
 * @param axis         归约轴（已归一化，-1 表示所有轴）
 * @param base_coords  基础坐标（归约轴位置填0）
 * @param user_data    用户数据指针
 * @param out_val      输出值指针（写入结果）
 */
typedef void (*reduce_inner_func)(const Tensor *x, int axis, const int *base_coords,
                                  void *user_data, float *out_val);

/**
 * @brief 通用归约主函数
 * @param x          输入张量
 * @param out        输出张量（已预先分配好形状）
 * @param axis       归约轴
 * @param keepdims   是否保留维度
 * @param inner_func 内层归约函数
 * @param user_data  传递给内层函数的用户数据
 * @return TensorStatus
 */
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

    // 确保输出独占数据（写时拷贝）
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

/* ==================== 内层归约函数（按类别分组） ==================== */

/* ---------- 基础归约：sum, mean, prod, max, min, argmax, argmin, any, all ---------- */

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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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

/* ---------- 方差与标准差 ---------- */

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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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

/* ---------- p 范数 ---------- */

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
    ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
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

/* ---------- 中位数 ---------- */

typedef struct
{
    float *work; // 临时工作数组
} median_data;

static void median_inner(const Tensor *x, int axis, const int *base_coords,
                         void *user_data, float *out_val)
{
    median_data *md = (median_data *)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    int n;
    if (axis == -1)
    {
        n = (int)x->size;
        for (int i = 0; i < n; i++)
            md->work[i] = x->data[i];
    }
    else
    {
        n = x->dims[axis];
        int inner_stride = x_strides[axis];
        ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
        for (int i = 0; i < n; i++)
            md->work[i] = x->data[base_off + i * inner_stride];
    }

    qsort(md->work, n, sizeof(float), compare_float);

    if (n % 2 == 1)
        *out_val = md->work[n / 2];
    else
        *out_val = (md->work[n / 2 - 1] + md->work[n / 2]) * 0.5f;
}

/* ---------- 众数 ---------- */

typedef struct
{
    float *work;
} mode_data;

static void mode_inner(const Tensor *x, int axis, const int *base_coords,
                       void *user_data, float *out_val)
{
    mode_data *md = (mode_data *)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    int n;
    if (axis == -1)
    {
        n = (int)x->size;
        for (int i = 0; i < n; i++)
            md->work[i] = x->data[i];
    }
    else
    {
        n = x->dims[axis];
        int inner_stride = x_strides[axis];
        ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
        for (int i = 0; i < n; i++)
            md->work[i] = x->data[base_off + i * inner_stride];
    }

    qsort(md->work, n, sizeof(float), compare_float);

    float best_val = md->work[0];
    int best_cnt = 1;
    int cur_cnt = 1;
    for (int i = 1; i < n; i++)
    {
        if (md->work[i] == md->work[i - 1])
        {
            cur_cnt++;
        }
        else
        {
            if (cur_cnt > best_cnt || (cur_cnt == best_cnt && md->work[i - 1] < best_val))
            {
                best_cnt = cur_cnt;
                best_val = md->work[i - 1];
            }
            cur_cnt = 1;
        }
    }
    // 最后一组
    if (cur_cnt > best_cnt || (cur_cnt == best_cnt && md->work[n - 1] < best_val))
        best_val = md->work[n - 1];

    *out_val = best_val;
}

/* ---------- 分位数 ---------- */

typedef struct
{
    float q;
    QuantileInterp interp;
    float *work;
} quantile_data;

static void quantile_inner(const Tensor *x, int axis, const int *base_coords,
                           void *user_data, float *out_val)
{
    quantile_data *qd = (quantile_data *)user_data;
    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);

    int n;
    if (axis == -1)
    {
        n = (int)x->size;
        for (int i = 0; i < n; i++)
            qd->work[i] = x->data[i];
    }
    else
    {
        n = x->dims[axis];
        int inner_stride = x_strides[axis];
        ptrdiff_t base_off = util_offset_from_coords(base_coords, x_strides, ndim);
        for (int i = 0; i < n; i++)
            qd->work[i] = x->data[base_off + i * inner_stride];
    }

    qsort(qd->work, n, sizeof(float), compare_float);

    *out_val = interpolate_quantile(qd->work, n, qd->q, qd->interp);
}

/* ==================== 对外 API ==================== */

/* ---------- 基础归约 ---------- */

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

/* ---------- 方差与标准差 ---------- */

TensorStatus tensor_var(const Tensor *x, Tensor *out, int axis, int keepdims, int unbiased)
{
    return reduce_op_general(x, out, axis, keepdims, var_inner, &unbiased);
}

TensorStatus tensor_std(const Tensor *x, Tensor *out, int axis, int keepdims, int unbiased)
{
    return reduce_op_general(x, out, axis, keepdims, std_inner, &unbiased);
}

/* ---------- p 范数 ---------- */

TensorStatus tensor_norm(const Tensor *x, Tensor *out, int axis, int keepdims, float p)
{
    return reduce_op_general(x, out, axis, keepdims, norm_inner, &p);
}

/* ---------- 中位数 ---------- */

TensorStatus tensor_median(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;

    int ax = (axis == -1) ? -1 : util_normalize_axis(axis, x->ndim);
    if (axis != -1 && ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    int max_block_size = (axis == -1) ? (int)x->size : x->dims[ax];
    float *work = (float *)malloc(max_block_size * sizeof(float));
    if (!work)
        return TENSOR_ERR_MEMORY;

    median_data md = {work};
    TensorStatus status = reduce_op_general(x, out, axis, keepdims, median_inner, &md);
    free(work);
    return status;
}

/* ---------- 众数 ---------- */

TensorStatus tensor_mode(const Tensor *x, Tensor *out, int axis, int keepdims)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;

    int ax = (axis == -1) ? -1 : util_normalize_axis(axis, x->ndim);
    if (axis != -1 && ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    int max_block_size = (axis == -1) ? (int)x->size : x->dims[ax];
    float *work = (float *)malloc(max_block_size * sizeof(float));
    if (!work)
        return TENSOR_ERR_MEMORY;

    mode_data md = {work};
    TensorStatus status = reduce_op_general(x, out, axis, keepdims, mode_inner, &md);
    free(work);
    return status;
}

/* ---------- 分位数 ---------- */

TensorStatus tensor_quantile(const Tensor *x, const Tensor *q, int axis, int keepdims,
                             QuantileInterp interp, Tensor *out)
{
    if (!x || !q || !out)
        return TENSOR_ERR_NULL_PTR;

    // 仅支持标量 q
    if (!(q->ndim == 0 || (q->ndim == 1 && q->dims[0] == 1)))
        return TENSOR_ERR_UNSUPPORTED;

    float qval = q->data[0];
    if (qval < 0.0f || qval > 1.0f)
        return TENSOR_ERR_INVALID_PARAM;

    int ax = (axis == -1) ? -1 : util_normalize_axis(axis, x->ndim);
    if (axis != -1 && ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    int max_block_size = (axis == -1) ? (int)x->size : x->dims[ax];
    float *work = (float *)malloc(max_block_size * sizeof(float));
    if (!work)
        return TENSOR_ERR_MEMORY;

    quantile_data qd = {qval, interp, work};
    TensorStatus status = reduce_op_general(x, out, axis, keepdims, quantile_inner, &qd);
    free(work);
    return status;
}

/* ==================== 累积操作实现 ==================== */

/**
 * @brief 通用累积操作框架
 * @param src          输入张量
 * @param axis         轴（已归一化）
 * @param dst          输出张量
 * @param init_val     初始值（如 -INFINITY 用于 cummax）
 * @param update_func  更新函数：给定当前累积值和新值，返回新累积值
 * @return TensorStatus
 */
static TensorStatus cum_op_general(const Tensor *src, int axis, Tensor *dst,
                                   float init_val,
                                   float (*update_func)(float accum, float new_val))
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;

    int ndim = src->ndim;
    int ax = util_normalize_axis(axis, ndim);
    if (ax < 0)
        return TENSOR_ERR_INVALID_PARAM;

    // 输出形状必须与输入相同
    if (dst->ndim != ndim || !util_shapes_equal(dst->dims, src->dims, ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(dst);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM], dst_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(dst, dst_strides);

    // 外部维度（除 axis 外）
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

        float accum = init_val;
        for (int j = 0; j < inner_len; ++j)
        {
            size_t src_off = src_base + j * inner_src_stride;
            float val = src->data[src_off];
            accum = update_func(accum, val);
            size_t dst_off = dst_base + j * inner_dst_stride;
            dst->data[dst_off] = accum;
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ---------- 更新函数 ---------- */

static float update_max(float accum, float new_val)
{
    return (accum > new_val) ? accum : new_val;
}

static float update_min(float accum, float new_val)
{
    return (accum < new_val) ? accum : new_val;
}

/**
 * @brief 稳定计算 log(exp(accum) + exp(new_val))
 */
static float update_logcumsumexp(float accum, float new_val)
{
    if (isinf(accum) && accum < 0) // accum == -INF
        return new_val;
    if (accum > new_val)
    {
        return accum + log1pf(expf(new_val - accum));
    }
    else
    {
        return new_val + log1pf(expf(accum - new_val));
    }
}

/* ---------- API ---------- */

TensorStatus tensor_cummax(const Tensor *src, int axis, Tensor *dst)
{
    return cum_op_general(src, axis, dst, -INFINITY, update_max);
}

TensorStatus tensor_cummin(const Tensor *src, int axis, Tensor *dst)
{
    return cum_op_general(src, axis, dst, INFINITY, update_min);
}

TensorStatus tensor_logcumsumexp(const Tensor *src, int axis, Tensor *dst)
{
    return cum_op_general(src, axis, dst, -INFINITY, update_logcumsumexp);
}