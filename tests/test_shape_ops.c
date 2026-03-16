#include "tensor.h"
#include "shape_ops.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")
#define EPS 1e-5f

static int approx_equal(float a, float b, float eps)
{
    if (isnan(a) && isnan(b))
        return 1;
    if (isinf(a) && isinf(b))
        return (a > 0) == (b > 0);
    return fabsf(a - b) < eps;
}

// static int check_tensor(const Tensor *t, const float *expected, size_t n)
// {
//     if (tensor_size(t) != n)
//         return 0;
//     for (size_t i = 0; i < n; ++i)
//         if (!approx_equal(t->data[i], expected[i], EPS))
//             return 0;
//     return 1;
// }

static int check_tensor(const Tensor *t, const float *expected, size_t n)
{
    if (tensor_size(t) != n)
        return 0;

    int ndim = t->ndim;
    if (ndim == 0)
    {
        return approx_equal(t->data[0], expected[0], EPS);
    }

    int *coords = (int *)malloc(ndim * sizeof(int));
    if (!coords)
        return 0;
    memset(coords, 0, ndim * sizeof(int));

    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);

    size_t idx = 0;
    while (1)
    {
        ptrdiff_t off = 0;
        for (int i = 0; i < ndim; ++i)
        {
            off += (ptrdiff_t)coords[i] * strides[i];
        }
        float val = t->data[off]; // 允许负偏移
        if (!approx_equal(val, expected[idx++], EPS))
        {
            free(coords);
            return 0;
        }
        if (util_increment_coords(coords, t->dims, ndim))
            break;
    }
    free(coords);
    return 1;
}

/* ---------- 测试 reshape ---------- */
void test_reshape()
{
    TEST("tensor_reshape");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_array(data, 2, dims);

    TensorStatus status = tensor_reshape(t, 1, (int[]){6});
    assert(status == TENSOR_OK);
    assert(tensor_ndim(t) == 1);
    assert(t->dims[0] == 6);
    assert(check_tensor(t, data, 6));

    status = tensor_reshape(t, 3, (int[]){2, 3, 1});
    assert(status == TENSOR_OK);
    assert(tensor_ndim(t) == 3);
    assert(t->dims[0] == 2 && t->dims[1] == 3 && t->dims[2] == 1);
    assert(check_tensor(t, data, 6));

    status = tensor_reshape(t, 2, (int[]){2, 2});
    assert(status == TENSOR_ERR_SHAPE_MISMATCH);

    tensor_destroy(t);
    PASS();
}

void test_reshape_view()
{
    TEST("tensor_reshape_view");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor dst = {0}; // 必须初始化为零

    TensorStatus status = tensor_reshape_view(src, 1, (int[]){6}, &dst);
    assert(status == TENSOR_OK);
    assert(dst.ndim == 1);
    assert(dst.dims[0] == 6);
    assert(dst.data == src->data);
    assert(*(dst.ref_count) == 2);
    assert(dst.strides[0] == 1);

    tensor_cleanup(&dst);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 flatten ---------- */
void test_flatten()
{
    TEST("tensor_flatten");
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++)
        data[i] = i;
    Tensor *src = tensor_from_array(data, 3, dims);
    Tensor dst = {0};

    tensor_flatten(src, 1, 2, &dst);
    assert(dst.ndim == 2);
    assert(dst.dims[0] == 2 && dst.dims[1] == 12);
    assert(dst.data == src->data);
    assert(*(dst.ref_count) == 2);
    assert(dst.data[23] == 23);

    tensor_cleanup(&dst);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 squeeze / unsqueeze ---------- */
void test_squeeze()
{
    TEST("tensor_squeeze");
    int dims[] = {1, 3, 1, 2};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data, 4, dims);
    Tensor dst = {0};

    tensor_squeeze(src, NULL, 0, &dst);
    assert(dst.ndim == 2);
    assert(dst.dims[0] == 3 && dst.dims[1] == 2);
    assert(check_tensor(&dst, data, 6));

    tensor_cleanup(&dst);
    dst = (Tensor){0};
    tensor_squeeze(src, (int[]){0}, 1, &dst);
    assert(dst.ndim == 3);
    assert(dst.dims[0] == 3 && dst.dims[1] == 1 && dst.dims[2] == 2);

    tensor_cleanup(&dst);
    tensor_destroy(src);
    PASS();
}

void test_unsqueeze()
{
    TEST("tensor_unsqueeze");
    int dims[] = {3, 2};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor dst = {0};

    tensor_unsqueeze(src, 0, &dst);
    assert(dst.ndim == 3);
    assert(dst.dims[0] == 1 && dst.dims[1] == 3 && dst.dims[2] == 2);
    assert(dst.data == src->data);

    tensor_cleanup(&dst);
    dst = (Tensor){0};
    tensor_unsqueeze(src, 2, &dst);
    assert(dst.ndim == 3);
    assert(dst.dims[0] == 3 && dst.dims[1] == 2 && dst.dims[2] == 1);

    tensor_cleanup(&dst);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 concat ---------- */
void test_concat()
{
    TEST("tensor_concat");
    int dims1[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    int dims2[] = {2, 3};
    float data2[] = {5, 6, 7, 8, 9, 10};
    Tensor *a = tensor_from_array(data1, 2, dims1);
    Tensor *b = tensor_from_array(data2, 2, dims2);
    const Tensor *inputs[] = {a, b};

    Tensor *out = tensor_create(2, (int[]){2, 5});
    tensor_concat(inputs, 2, 1, out);
    float expected[] = {1, 2, 5, 6, 7, 3, 4, 8, 9, 10};
    assert(check_tensor(out, expected, 10));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 stack ---------- */
void test_stack()
{
    TEST("tensor_stack");
    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};
    Tensor *a = tensor_from_array(data1, 2, dims);
    Tensor *b = tensor_from_array(data2, 2, dims);
    const Tensor *inputs[] = {a, b};

    Tensor *out = tensor_create(3, (int[]){2, 2, 2});
    tensor_stack(inputs, 2, 1, out);
    float expected[] = {1, 2, 5, 6, 3, 4, 7, 8};
    assert(check_tensor(out, expected, 8));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 split ---------- */
void test_split()
{
    TEST("tensor_split");
    int dims[] = {2, 6};
    float data[12];
    for (int i = 0; i < 12; i++)
        data[i] = i;
    Tensor *src = tensor_from_array(data, 2, dims);
    int sizes[] = {2, 4};
    Tensor *outputs[2];

    tensor_split(src, 1, sizes, 2, outputs);
    assert(outputs[0]->ndim == 2);
    assert(outputs[0]->dims[1] == 2);
    assert(outputs[1]->dims[1] == 4);

    float expected0[] = {0, 1, 6, 7};
    float expected1[] = {2, 3, 4, 5, 8, 9, 10, 11};
    assert(check_tensor(outputs[0], expected0, 4));
    assert(check_tensor(outputs[1], expected1, 8));

    tensor_destroy(outputs[0]);
    tensor_destroy(outputs[1]);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 slice ---------- */
void test_slice()
{
    TEST("tensor_slice");
    int dims[] = {3, 4};
    float data[12];
    for (int i = 0; i < 12; i++)
        data[i] = i;
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor dst = {0};

    int starts[] = {0, 1};
    int ends[] = {3, 3};
    int steps[] = {1, 2};
    tensor_slice(src, starts, ends, steps, &dst);
    float expected[] = {1, 5, 9};
    assert(check_tensor(&dst, expected, 3));

    tensor_cleanup(&dst);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 repeat ---------- */
void test_repeat()
{
    TEST("tensor_repeat");
    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, (int[]){2, 4});

    tensor_repeat(src, 1, 2, out);
    float expected[] = {1, 1, 2, 2, 3, 3, 4, 4};
    assert(check_tensor(out, expected, 8));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 tile ---------- */
void test_tile()
{
    TEST("tensor_tile");
    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, (int[]){4, 4});

    int reps[] = {2, 2};
    tensor_tile(src, reps, out);
    float expected[] = {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4};
    assert(check_tensor(out, expected, 16));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 transpose_axes ---------- */
void test_transpose_axes()
{
    TEST("tensor_transpose_axes");
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++)
        data[i] = i;
    Tensor *src = tensor_from_array(data, 3, dims);
    Tensor dst = {0};
    dst.dims = (int[]){2, 4, 3}; // 需要预先指定输出形状（函数会验证）
    dst.ndim = 3;

    int axes[] = {0, 2, 1};
    tensor_transpose_axes(src, axes, &dst);
    assert(dst.ndim == 3);
    assert(dst.dims[0] == 2 && dst.dims[1] == 4 && dst.dims[2] == 3);

    size_t src_off = 1 * 12 + 2 * 4 + 3;
    size_t dst_off = 1 * 12 + 3 * 3 + 2;
    assert(dst.data[dst_off] == src->data[src_off]);

    // 注意：这里 dst 是视图，需要销毁时释放其内部 dims/strides（由 tensor_transpose_axes 分配）
    tensor_cleanup(&dst);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 swapaxes ---------- */
void test_swapaxes()
{
    TEST("tensor_swapaxes");
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++)
        data[i] = i;
    Tensor *t = tensor_from_array(data, 3, dims);

    tensor_swapaxes(t, 0, 2);
    assert(t->dims[0] == 4 && t->dims[1] == 3 && t->dims[2] == 2);
    assert(t->data[23] == 23);

    tensor_destroy(t);
    PASS();
}

/* ---------- 测试 flip ---------- */
void test_flip()
{
    TEST("tensor_flip");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor dst = {0};
    int out_dims[] = {2, 3}; // 与输入形状相同
    dst.dims = out_dims;
    dst.ndim = 2;

    int axes[] = {1};
    TensorStatus status = tensor_flip(src, axes, 1, &dst);
    assert(status == TENSOR_OK); // 确保函数成功
    float expected[] = {3, 2, 1, 6, 5, 4};
    assert(check_tensor(&dst, expected, 6));

    tensor_cleanup(&dst);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 pad ---------- */
void test_pad()
{
    TEST("tensor_pad");
    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, (int[]){4, 4});

    int pad_widths[] = {1, 1, 1, 1};
    tensor_pad(src, pad_widths, PAD_CONSTANT, 0, out);
    float expected[] = {
        0, 0, 0, 0,
        0, 1, 2, 0,
        0, 3, 4, 0,
        0, 0, 0, 0};
    assert(check_tensor(out, expected, 16));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 cumsum ---------- */
void test_cumsum()
{
    TEST("tensor_cumsum");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_cumsum(src, 1, out);
    float expected[] = {1, 3, 6, 4, 9, 15};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 cumprod ---------- */
void test_cumprod()
{
    TEST("tensor_cumprod");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_cumprod(src, 1, out);
    float expected[] = {1, 2, 6, 4, 20, 120};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}
/* ---------- 测试 pad 的其他模式 ---------- */
void test_pad_reflect()
{
    TEST("tensor_pad reflect");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *src = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, (int[]){5}); // 左2右0

    int pad_widths[] = {2, 0};
    tensor_pad(src, pad_widths, PAD_REFLECT, 0, out);
    float expected[] = {3, 2, 1, 2, 3}; // 反射: [2,1]? 根据前面分析，左2应取 [3,2]
    assert(check_tensor(out, expected, 5));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_pad_replicate()
{
    TEST("tensor_pad replicate");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *src = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, (int[]){5}); // 左2右0

    int pad_widths[] = {2, 0};
    tensor_pad(src, pad_widths, PAD_REPLICATE, 0, out);
    float expected[] = {1, 1, 1, 2, 3}; // 复制边缘: 左边填充第一个元素
    assert(check_tensor(out, expected, 5));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_pad_circular()
{
    TEST("tensor_pad circular");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *src = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, (int[]){5}); // 左2右0

    int pad_widths[] = {2, 0};
    tensor_pad(src, pad_widths, PAD_CIRCULAR, 0, out);
    float expected[] = {2, 3, 1, 2, 3}; // 循环: 左2取末尾两个 [2,3]
    assert(check_tensor(out, expected, 5));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_pad_2d_reflect()
{
    TEST("tensor_pad 2D reflect");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, (int[]){4, 5}); // 上下各1，左右各1

    int pad_widths[] = {1, 1, 1, 1};
    tensor_pad(src, pad_widths, PAD_REFLECT, 0, out);
    float expected[] = {
        5, 4, 5, 6, 5,
        2, 1, 2, 3, 2,
        5, 4, 5, 6, 5,
        2, 1, 2, 3, 2};
    assert(check_tensor(out, expected, 20));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}
/* ---------- 主函数 ---------- */
int main()
{
    test_reshape();
    test_reshape_view();
    test_flatten();
    test_squeeze();
    test_unsqueeze();
    test_concat();
    test_stack();
    test_split();
    test_slice();
    test_repeat();
    test_tile();
    test_transpose_axes();
    test_swapaxes();
    test_flip();
    test_pad();
    test_cumsum();
    test_cumprod();
    test_pad_reflect();
    test_pad_replicate();
    test_pad_circular();
    test_pad_2d_reflect();

    printf("All shape_ops tests passed!\n");
    return 0;
}