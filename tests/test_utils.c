#include "tensor.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
static void test_broadcast_shape()
{
    int dims1[] = {2, 3};
    int dims2[] = {3};
    int out_dims[2];
    int out_ndim;
    int ok = util_broadcast_shape(dims1, 2, dims2, 1, out_dims, &out_ndim);
    assert(ok == 1);
    assert(out_ndim == 2);
    assert(out_dims[0] == 2 && out_dims[1] == 3);

    dims2[0] = 1;
    ok = util_broadcast_shape(dims1, 2, dims2, 1, out_dims, &out_ndim);
    assert(ok == 1);
    assert(out_ndim == 2);
    assert(out_dims[0] == 2 && out_dims[1] == 3);

    dims2[0] = 2; // 不兼容
    ok = util_broadcast_shape(dims1, 2, dims2, 1, out_dims, &out_ndim);
    assert(ok == 0);

    printf("test_broadcast_shape passed\n");
}

static void test_fill_padded_strides()
{
    int dims[] = {2, 3};
    int strides[] = {3, 1};
    Tensor t;
    t.ndim = 2;
    t.dims = dims;
    t.strides = strides;
    t.owns_dims_strides = 0;

    int out_ndim = 3;
    int out_dims[] = {2, 3, 4};
    int padded[3];
    util_fill_padded_strides(&t, out_ndim, out_dims, padded);
    assert(padded[0] == 0);
    assert(padded[1] == 3);
    assert(padded[2] == 1);

    // 连续张量（strides = NULL）
    t.strides = NULL;
    util_fill_padded_strides(&t, out_ndim, out_dims, padded);
    assert(padded[0] == 0);
    // 连续步长应为：最后一个维度 1，向前依次乘
    // 原形状(2,3)连续步长应为[3,1]
    assert(padded[1] == 3);
    assert(padded[2] == 1);

    printf("test_fill_padded_strides passed\n");
}

static void test_get_effective_strides()
{
    int dims[] = {2, 3, 4};
    int strides[] = {12, 4, 1};
    Tensor t;
    t.ndim = 3;
    t.dims = dims;
    t.strides = strides;
    t.owns_dims_strides = 0;

    int out[3];
    util_get_effective_strides(&t, out);
    assert(out[0] == 12 && out[1] == 4 && out[2] == 1);

    t.strides = NULL;
    util_get_effective_strides(&t, out);
    assert(out[0] == 12 && out[1] == 4 && out[2] == 1); // 连续

    printf("test_get_effective_strides passed\n");
}

static void test_normalize_axis()
{
    assert(util_normalize_axis(0, 3) == 0);
    assert(util_normalize_axis(2, 3) == 2);
    assert(util_normalize_axis(-1, 3) == 2);
    assert(util_normalize_axis(-3, 3) == 0);
    assert(util_normalize_axis(3, 3) == -1);
    assert(util_normalize_axis(-4, 3) == -1);
    printf("test_normalize_axis passed\n");
}

static void test_offset_from_coords()
{
    int strides[] = {12, 4, 1};
    int coords[] = {1, 2, 3};
    size_t off = util_offset_from_coords(coords, strides, 3);
    assert(off == 1 * 12 + 2 * 4 + 3 * 1);
    printf("test_offset_from_coords passed\n");
}

// 用于测试的简单操作函数
static float neg(float x) { return -x; }
static float add(float a, float b) { return a + b; }
static float add3(float a, float b, float c) { return a + b + c; }

static void test_unary_op_general()
{
    int dims[] = {2, 3};
    Tensor *x = tensor_from_array((float[]){1, 2, 3, 4, 5, 6}, 2, dims);
    Tensor *out = tensor_create(2, dims);
    TensorStatus st = util_unary_op_general(x, out, neg);
    assert(st == TENSOR_OK);
    float expected[] = {-1, -2, -3, -4, -5, -6};
    assert(memcmp(out->data, expected, 6 * sizeof(float)) == 0);
    tensor_destroy(x);
    tensor_destroy(out);
    printf("test_unary_op_general passed\n");
}

static void test_binary_op_general()
{
    int dims_a[] = {2, 3};
    int dims_b[] = {3};
    Tensor *a = tensor_from_array((float[]){1, 2, 3, 4, 5, 6}, 2, dims_a);
    Tensor *b = tensor_from_array((float[]){10, 20, 30}, 1, dims_b);
    int out_dims[] = {2, 3};
    Tensor *out = tensor_create(2, out_dims);
    TensorStatus st = util_binary_op_general(a, b, out, add);
    assert(st == TENSOR_OK);
    float expected[] = {11, 22, 33, 14, 25, 36};
    assert(memcmp(out->data, expected, 6 * sizeof(float)) == 0);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    printf("test_binary_op_general passed\n");
}

static void test_ternary_op_general()
{
    int dims[] = {2, 2};
    Tensor *a = tensor_from_array((float[]){1, 2, 3, 4}, 2, dims);
    Tensor *b = tensor_from_array((float[]){10, 20}, 1, (int[]){2});
    Tensor *c = tensor_from_array((float[]){100, 200}, 1, (int[]){2});
    int out_dims[] = {2, 2};
    Tensor *out = tensor_create(2, out_dims);
    TensorStatus st = util_ternary_op_general(a, b, c, out, add3);
    assert(st == TENSOR_OK);
    float expected[] = {1 + 10 + 100, 2 + 20 + 200, 3 + 10 + 100, 4 + 20 + 200}; // 广播 b 和 c 到 [2,2]
    for (int i = 0; i < 4; ++i)
        assert(fabsf(out->data[i] - expected[i]) < 1e-5);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(c);
    tensor_destroy(out);
    printf("test_ternary_op_general passed\n");
}

static void test_save_load()
{
    int dims[] = {2, 3};
    Tensor *t = tensor_from_array((float[]){1, 2, 3, 4, 5, 6}, 2, dims);
    assert(tensor_save(t, "test_tensor.bin") == TENSOR_OK);
    Tensor *t2 = NULL;
    assert(tensor_load(&t2, "test_tensor.bin") == TENSOR_OK);
    assert(t2->ndim == 2);
    assert(t2->dims[0] == 2 && t2->dims[1] == 3);
    assert(memcmp(t->data, t2->data, 6 * sizeof(float)) == 0);
    tensor_destroy(t);
    tensor_destroy(t2);
    remove("test_tensor.bin");
    printf("test_save_load passed\n");
}

static void test_allclose()
{
    int dims[] = {3};
    Tensor *a = tensor_from_array((float[]){1.0, 2.0, 3.0}, 1, dims);
    Tensor *b = tensor_from_array((float[]){1.0, 2.0, 3.0001}, 1, dims);
    assert(tensor_allclose(a, b, 1e-3, 1e-5) == 1); // rtol足够
    assert(tensor_allclose(a, b, 1e-6, 1e-5) == 0); // 太小
    tensor_destroy(a);
    tensor_destroy(b);
    printf("test_allclose passed\n");
}

static void test_has_nan_inf()
{
    int dims[] = {3};
    float data[] = {1.0, NAN, 3.0};
    Tensor *t = tensor_from_array(data, 1, dims);
    assert(tensor_has_nan(t) == 1);
    assert(tensor_has_inf(t) == 0);
    data[1] = INFINITY;
    memcpy(t->data, data, 3 * sizeof(float));
    assert(tensor_has_nan(t) == 0);
    assert(tensor_has_inf(t) == 1);
    tensor_destroy(t);
    printf("test_has_nan_inf passed\n");
}

static void test_fill_init()
{
    int dims[] = {2, 2};
    Tensor *t = tensor_create(2, dims);
    tensor_fill(t, 5.0f);
    for (size_t i = 0; i < 4; ++i)
        assert(t->data[i] == 5.0f);

    tensor_uniform_init(t, -1, 1);
    for (size_t i = 0; i < 4; ++i)
        assert(t->data[i] >= -1 && t->data[i] <= 1);

    tensor_normal_init(t, 0, 1);
    // 不验证具体值，只确保无异常

    tensor_xavier_init(t, 3, 3);
    float scale = sqrtf(6.0f / 6.0f); // 1.0
    for (size_t i = 0; i < 4; ++i)
        assert(t->data[i] >= -1 && t->data[i] <= 1);

    tensor_destroy(t);
    printf("test_fill_init passed\n");
}

int main()
{
    test_broadcast_shape();
    test_fill_padded_strides();
    test_get_effective_strides();
    test_normalize_axis();
    test_offset_from_coords();
    test_unary_op_general();
    test_binary_op_general();
    test_ternary_op_general();
    test_save_load();
    test_allclose();
    test_has_nan_inf();
    test_fill_init();
    printf("All utils tests passed!\n");
    return 0;
}