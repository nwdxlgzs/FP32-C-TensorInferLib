#include "tensor.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

/* ==================== 调试工具 ==================== */

void tensor_print(const Tensor *t, const char *name, int max_elements)
{
    if (!t)
    {
        printf("(null tensor)\n");
        return;
    }

    if (name)
        printf("%s: ", name);
    printf("ndim=%d, size=%zu, [", t->ndim, t->size);
    for (int i = 0; i < t->ndim; ++i)
    {
        if (i > 0)
            printf(", ");
        printf("%d", t->dims[i]);
    }
    printf("]\n");

    if (max_elements == 0)
        return;

    size_t n = (max_elements < 0 || (size_t)max_elements > t->size) ? t->size : (size_t)max_elements;
    printf("[");
    for (size_t i = 0; i < n; ++i)
    {
        if (i > 0)
            printf(", ");
        printf("%g", t->data[i]);
    }
    if (n < t->size)
        printf(", ...");
    printf("]\n");
}

TensorStatus tensor_save(const Tensor *t, const char *filename)
{
    if (!t || !filename)
        return TENSOR_ERR_NULL_PTR;

    // 如果 t 不连续，创建一个连续副本
    Tensor *cont = NULL;
    if (!util_is_contiguous(t))
    {
        cont = tensor_clone(t); // clone 创建连续张量
        if (!cont)
            return TENSOR_ERR_MEMORY;
        t = cont;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        if (cont)
            tensor_destroy(cont);
        return TENSOR_ERR_INVALID_PARAM;
    }

    if (fwrite(&t->ndim, sizeof(int), 1, fp) != 1)
        goto error;
    if (t->ndim > 0 && fwrite(t->dims, sizeof(int), t->ndim, fp) != (size_t)t->ndim)
        goto error;
    if (fwrite(t->data, sizeof(float), t->size, fp) != t->size)
        goto error;

    fclose(fp);
    if (cont)
        tensor_destroy(cont);
    return TENSOR_OK;

error:
    fclose(fp);
    if (cont)
        tensor_destroy(cont);
    return TENSOR_ERR_MEMORY;
}

TensorStatus tensor_load(Tensor **t, const char *filename)
{
    if (!t || !filename)
        return TENSOR_ERR_NULL_PTR;

    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return TENSOR_ERR_INVALID_PARAM;

    int ndim;
    if (fread(&ndim, sizeof(int), 1, fp) != 1)
        goto error;

    int dims[TENSOR_MAX_DIM];
    if (ndim > 0)
    {
        if (fread(dims, sizeof(int), ndim, fp) != (size_t)ndim)
            goto error;
    }

    *t = tensor_create(ndim, ndim > 0 ? dims : NULL);
    if (!*t)
        goto error;

    if (fread((*t)->data, sizeof(float), (*t)->size, fp) != (*t)->size)
    {
        tensor_destroy(*t);
        *t = NULL;
        goto error;
    }

    fclose(fp);
    return TENSOR_OK;

error:
    fclose(fp);
    return TENSOR_ERR_MEMORY;
}

int tensor_allclose(const Tensor *a, const Tensor *b, float rtol, float atol)
{
    if (!a || !b)
        return 0;
    if (a->ndim != b->ndim || a->size != b->size)
        return 0;
    for (int i = 0; i < a->ndim; ++i)
        if (a->dims[i] != b->dims[i])
            return 0;

    for (size_t i = 0; i < a->size; ++i)
    {
        float va = a->data[i];
        float vb = b->data[i];
        if (isnan(va) || isnan(vb))
        {
            if (!(isnan(va) && isnan(vb)))
                return 0;
        }
        else
        {
            float diff = fabsf(va - vb);
            float tol = atol + rtol * fabsf(vb);
            if (diff > tol)
                return 0;
        }
    }
    return 1;
}

int tensor_has_nan(const Tensor *t)
{
    if (!t)
        return 0;
    for (size_t i = 0; i < t->size; ++i)
        if (isnan(t->data[i]))
            return 1;
    return 0;
}

int tensor_has_inf(const Tensor *t)
{
    if (!t)
        return 0;
    for (size_t i = 0; i < t->size; ++i)
        if (isinf(t->data[i]))
            return 1;
    return 0;
}

TensorStatus tensor_fill(Tensor *t, float value)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    TensorStatus status = tensor_make_unique(t);
    if (status != TENSOR_OK)
        return status;

    int ndim = t->ndim;
    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);
    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t off = util_offset_from_coords(coords, strides, ndim);
        t->data[off] = value;
        if (util_increment_coords(coords, t->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_normal_init(Tensor *t, float mean, float std)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    TensorStatus status = tensor_make_unique(t);
    if (status != TENSOR_OK)
        return status;

    int ndim = t->ndim;
    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);
    int coords[TENSOR_MAX_DIM] = {0};

    int have_spare = 0;
    float spare;
    while (1)
    {
        size_t off = util_offset_from_coords(coords, strides, ndim);
        if (have_spare)
        {
            t->data[off] = mean + std * spare;
            have_spare = 0;
        }
        else
        {
            float u1, u2, r, theta;
            do
            {
                u1 = (float)rand() / (RAND_MAX + 1.0f);
            } while (u1 <= FLT_MIN);
            u2 = (float)rand() / (RAND_MAX + 1.0f);
            r = sqrtf(-2.0f * logf(u1));
            theta = 2.0f * (float)M_PI * u2;
            float z0 = r * cosf(theta);
            float z1 = r * sinf(theta);
            t->data[off] = mean + std * z0;
            spare = z1;
            have_spare = 1;
        }

        if (util_increment_coords(coords, t->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_uniform_init(Tensor *t, float low, float high)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    TensorStatus status = tensor_make_unique(t);
    if (status != TENSOR_OK)
        return status;

    float range = high - low;
    int ndim = t->ndim;
    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);
    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t off = util_offset_from_coords(coords, strides, ndim);
        float r = (float)rand() / RAND_MAX;
        t->data[off] = low + r * range;
        if (util_increment_coords(coords, t->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_xavier_init(Tensor *t, int fan_in, int fan_out)
{
    float scale = sqrtf(6.0f / (fan_in + fan_out));
    return tensor_uniform_init(t, -scale, scale);
}

int util_broadcast_shapes(const int *dims[], const int ndims[], int num_tensors,
                          int *out_dims, int *out_ndim)
{
    if (num_tensors == 0)
    {
        *out_ndim = 0;
        return 1;
    }

    // 以第一个形状为基准
    *out_ndim = ndims[0];
    for (int i = 0; i < ndims[0]; ++i)
        out_dims[i] = dims[0][i];

    for (int k = 1; k < num_tensors; ++k)
    {
        int temp_ndim;
        int temp_dims[TENSOR_MAX_DIM];
        if (!util_broadcast_shape(out_dims, *out_ndim,
                                  dims[k], ndims[k],
                                  temp_dims, &temp_ndim))
            return 0; // 广播失败
        *out_ndim = temp_ndim;
        for (int i = 0; i < temp_ndim; ++i)
            out_dims[i] = temp_dims[i];
    }
    return 1;
}

int util_broadcast_shape(const int *dims_a, int ndim_a,
                         const int *dims_b, int ndim_b,
                         int *out_dims, int *out_ndim)
{
    *out_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    int i_a = ndim_a - 1;
    int i_b = ndim_b - 1;
    for (int i = *out_ndim - 1; i >= 0; --i)
    {
        int da = (i_a >= 0) ? dims_a[i_a] : 1;
        int db = (i_b >= 0) ? dims_b[i_b] : 1;
        if (da != db && da != 1 && db != 1)
            return 0;
        out_dims[i] = (da > db) ? da : db;
        --i_a;
        --i_b;
    }
    return 1;
}

void util_fill_padded_strides(const Tensor *t, int out_ndim,
                              const int *out_dims, int *padded_strides)
{
    int offset = out_ndim - t->ndim;
    for (int i = 0; i < offset; ++i)
        padded_strides[i] = 0;

    if (t->strides)
    {
        for (int i = 0; i < t->ndim; ++i)
        {
            int stride = t->strides[i];
            if (t->dims[i] == 1)
                stride = 0;
            padded_strides[offset + i] = stride;
        }
    }
    else
    {
        int stride = 1;
        for (int i = t->ndim - 1; i >= 0; --i)
        {
            if (t->dims[i] == 1)
            {
                padded_strides[offset + i] = 0;
            }
            else
            {
                padded_strides[offset + i] = stride;
                stride *= t->dims[i];
            }
        }
    }
}

void util_get_effective_strides(const Tensor *t, int *strides_out)
{
    if (t->strides)
    {
        memcpy(strides_out, t->strides, t->ndim * sizeof(int));
    }
    else
    {
        int stride = 1;
        for (int i = t->ndim - 1; i >= 0; --i)
        {
            strides_out[i] = stride;
            stride *= t->dims[i];
        }
    }
}

size_t util_offset_from_coords(const int *coords, const int *strides, int ndim)
{
    size_t off = 0;
    for (int i = 0; i < ndim; ++i)
        off += coords[i] * strides[i];
    return off;
}

/* ==================== 通用迭代器 ==================== */

TensorStatus util_unary_op_general(const Tensor *x, Tensor *out, UnaryOp op)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; ++i)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};

    while (1)
    {
        size_t x_off = 0, out_off = 0;
        for (int i = 0; i < ndim; ++i)
        {
            x_off += coords[i] * x_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        out->data[out_off] = op(x->data[x_off]);
        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus util_binary_op_general(const Tensor *a, const Tensor *b,
                                    Tensor *out, BinaryOp op)
{
    if (!a || !b || !out)
        return TENSOR_ERR_NULL_PTR;

    int out_ndim;
    int out_dims[TENSOR_MAX_DIM];
    if (!util_broadcast_shape(a->dims, a->ndim, b->dims, b->ndim, out_dims, &out_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; ++i)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    if (!util_broadcast_shape(a->dims, a->ndim, b->dims, b->ndim, out_dims, &out_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;
    // 验证 out 形状与原代码相同（略）

    // 获取广播后的步长
    int a_strides[TENSOR_MAX_DIM], b_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_fill_padded_strides(a, out_ndim, out_dims, a_strides);
    util_fill_padded_strides(b, out_ndim, out_dims, b_strides);
    util_get_effective_strides(out, out_strides);

    // 判断输出是否连续（行主序）
    int out_contiguous = util_is_contiguous(out);
    if (out_contiguous)
    {
        // 输出连续，采用一维线性遍历
        size_t total = out->size;
        // 预先计算每个输入的“扁平化”步长？实际上，对于连续输出，我们可以直接按线性顺序遍历，
        // 但输入偏移量需要根据当前线性索引对应的坐标来计算。
        // 为了快速从线性索引得到坐标，可以预先计算每个维度的累乘因子。
        int stride_mult[TENSOR_MAX_DIM]; // 每个维度上的累乘因子，用于从线性索引提取坐标
        stride_mult[out_ndim - 1] = 1;
        for (int i = out_ndim - 2; i >= 0; --i)
        {
            stride_mult[i] = stride_mult[i + 1] * out_dims[i + 1];
        }

        for (size_t idx = 0; idx < total; ++idx)
        {
            // 从线性索引计算坐标（除法和取模）
            int coords[TENSOR_MAX_DIM];
            size_t rem = idx;
            for (int i = 0; i < out_ndim; ++i)
            {
                coords[i] = rem / stride_mult[i];
                rem %= stride_mult[i];
            }
            // 计算各输入偏移
            size_t a_off = 0, b_off = 0;
            for (int i = 0; i < out_ndim; ++i)
            {
                a_off += coords[i] * a_strides[i];
                b_off += coords[i] * b_strides[i];
            }
            out->data[idx] = op(a->data[a_off], b->data[b_off]);
        }
    }
    else
    {
        // 输出不连续，仍使用坐标递增遍历
        int coords[TENSOR_MAX_DIM] = {0};
        while (1)
        {
            size_t a_off = 0, b_off = 0, out_off = 0;
            for (int i = 0; i < out_ndim; ++i)
            {
                a_off += coords[i] * a_strides[i];
                b_off += coords[i] * b_strides[i];
                out_off += coords[i] * out_strides[i];
            }
            out->data[out_off] = op(a->data[a_off], b->data[b_off]);
            if (util_increment_coords(coords, out_dims, out_ndim))
                break;
        }
    }
    return TENSOR_OK;
}

TensorStatus util_ternary_op_general(const Tensor *a, const Tensor *b,
                                     const Tensor *c, Tensor *out, TernaryOp op)
{
    if (!a || !b || !c || !out)
        return TENSOR_ERR_NULL_PTR;

    // 先广播 a 和 b
    int tmp_ndim;
    int tmp_dims[TENSOR_MAX_DIM];
    if (!util_broadcast_shape(a->dims, a->ndim, b->dims, b->ndim, tmp_dims, &tmp_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    int out_ndim;
    int out_dims[TENSOR_MAX_DIM];
    if (!util_broadcast_shape(tmp_dims, tmp_ndim, c->dims, c->ndim, out_dims, &out_ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; ++i)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int a_strides[TENSOR_MAX_DIM], b_strides[TENSOR_MAX_DIM];
    int c_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_fill_padded_strides(a, out_ndim, out_dims, a_strides);
    util_fill_padded_strides(b, out_ndim, out_dims, b_strides);
    util_fill_padded_strides(c, out_ndim, out_dims, c_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};

    while (1)
    {
        size_t a_off = 0, b_off = 0, c_off = 0, out_off = 0;
        for (int i = 0; i < out_ndim; ++i)
        {
            a_off += coords[i] * a_strides[i];
            b_off += coords[i] * b_strides[i];
            c_off += coords[i] * c_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        out->data[out_off] = op(a->data[a_off], b->data[b_off], c->data[c_off]);
        if (util_increment_coords(coords, out_dims, out_ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus util_binary_op_scalar(const Tensor *a, float scalar,
                                   Tensor *out, BinaryOp op)
{
    Tensor scalar_tensor;
    scalar_tensor.data = &scalar;
    scalar_tensor.ndim = 0;
    scalar_tensor.dims = NULL;
    scalar_tensor.strides = NULL;
    scalar_tensor.size = 1;
    scalar_tensor.ref_count = NULL;
    scalar_tensor.owns_dims_strides = 0;
    return util_binary_op_general(a, &scalar_tensor, out, op);
}
TensorStatus util_generate_op(Tensor *t, float (*gen)(void *), void *user_data)
{
    if (!t || !gen)
        return TENSOR_ERR_NULL_PTR;

    // 确保独占数据
    TensorStatus status = tensor_make_unique(t);
    if (status != TENSOR_OK)
        return status;

    int ndim = t->ndim;
    if (ndim == 0)
    { // 标量
        t->data[0] = gen(user_data);
        return TENSOR_OK;
    }

    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);

    int coords[TENSOR_MAX_DIM] = {0};
    const int *dims = t->dims;

    while (1)
    {
        size_t off = util_offset_from_coords(coords, strides, ndim);
        t->data[off] = gen(user_data);

        if (util_increment_coords(coords, dims, ndim))
            break;
    }
    return TENSOR_OK;
}
/* 检查张量是否连续（行主序） */
int util_is_contiguous(const Tensor *t)
{
    if (!t)
        return 0;
    if (t->strides == NULL)
        return 1; // 显式标记为连续
    if (t->ndim == 0)
        return 1;
    int expected = 1;
    for (int i = t->ndim - 1; i >= 0; --i)
    {
        if (t->strides[i] != expected)
            return 0;
        expected *= t->dims[i];
    }
    return 1;
}

/* 复制整数数组 */
int *util_copy_ints(const int *src, int n)
{
    if (n <= 0)
        return NULL;
    int *dst = (int *)malloc(n * sizeof(int));
    if (dst)
        memcpy(dst, src, n * sizeof(int));
    return dst;
}

/* 计算连续存储的步长（行主序） */
int *util_calc_contiguous_strides(const int *dims, int ndim)
{
    if (ndim == 0)
        return NULL;
    int *strides = (int *)malloc(ndim * sizeof(int));
    if (!strides)
        return NULL;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * dims[i + 1];
    return strides;
}

/* 计算元素总数 */
size_t util_calc_size(const int *dims, int ndim)
{
    if (ndim <= 0)
        return 1; // 标量情况
    size_t size = 1;
    for (int i = 0; i < ndim; ++i)
        size *= dims[i];
    return size;
}
/* 检查两个形状是否相等 */
int util_shapes_equal(const int *a, const int *b, int ndim)
{
    for (int i = 0; i < ndim; ++i)
        if (a[i] != b[i])
            return 0;
    return 1;
}
/* 归一化轴索引（支持负索引） */
int util_normalize_axis(int axis, int ndim)
{
    if (axis < 0)
        axis += ndim;
    if (axis < 0 || axis >= ndim)
        return -1;
    return axis;
}

int util_increment_coords(int *coords, const int *dims, int ndim)
{
    int axis = ndim - 1;
    while (axis >= 0 && ++coords[axis] == dims[axis])
    {
        coords[axis] = 0;
        --axis;
    }
    return (axis < 0) ? 1 : 0;
}

void util_coords_from_linear(size_t linear, const int *dims, int ndim, int *coords)
{
    size_t remaining = linear;
    for (int i = ndim - 1; i >= 0; --i)
    {
        int dim = dims[i];
        coords[i] = remaining % dim;
        remaining /= dim;
    }
    // remaining 最终应为 0，若不为0则说明 linear 超出总大小（调用者应保证合法）
}

int util_same_data(const Tensor *a, const Tensor *b)
{
    if (!a || !b)
        return 0;

    // 如果数据指针相同，肯定共享
    if (a->data == b->data)
        return 1;

    // 如果都有引用计数器且指向同一计数器，则属于同一数据块
    if (a->ref_count && b->ref_count && a->ref_count == b->ref_count)
        return 1;

    // 其他情况：不共享
    return 0;
}

void util_clear_tensor(Tensor *t)
{
    int ndim = t->ndim;
    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);
    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t off = util_offset_from_coords(coords, strides, ndim);
        t->data[off] = 0.0f;
        if (util_increment_coords(coords, t->dims, ndim))
            break;
    }
}

void tensor_print_logical(const Tensor *t, const char *name, int max_elements)
{
    if (!t)
    {
        printf("(null tensor)\n");
        return;
    }

    if (name)
        printf("%s: ", name);
    printf("ndim=%d, size=%zu, [", t->ndim, t->size);
    for (int i = 0; i < t->ndim; ++i)
    {
        if (i > 0)
            printf(", ");
        printf("%d", t->dims[i]);
    }
    printf("]\n");

    if (max_elements == 0)
        return;

    size_t n = (max_elements < 0 || (size_t)max_elements > t->size) ? t->size : (size_t)max_elements;
    int ndim = t->ndim;
    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);

    int coords[TENSOR_MAX_DIM] = {0};
    size_t printed = 0;
    printf("[");
    while (printed < n)
    {
        size_t off = util_offset_from_coords(coords, strides, ndim);
        if (printed > 0)
            printf(", ");
        printf("%g", t->data[off]);
        printed++;
        if (util_increment_coords(coords, t->dims, ndim))
            break;
    }
    if (printed < t->size)
        printf(", ...");
    printf("]\n");
}