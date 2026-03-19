#include "tensor.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* 辅助宏 */
#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")

/* ==================== 已有测试（保留） ==================== */

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
    ptrdiff_t off = util_offset_from_coords(coords, strides, 3);
    assert(off == 1 * 12 + 2 * 4 + 3 * 1);
    printf("test_offset_from_coords passed\n");
}

// 用于测试的简单操作函数
static float neg(float x, void *user_data) { return -x; }
static float add(float a, float b, void *user_data) { return a + b; }
static float add3(float a, float b, float c, void *user_data) { return a + b + c; }

static void test_unary_op_general()
{
    int dims[] = {2, 3};
    Tensor *x = tensor_from_array((float[]){1, 2, 3, 4, 5, 6}, 2, dims);
    Tensor *out = tensor_create(2, dims);
    TensorStatus st = util_unary_op_general_NUD(x, out, neg);
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
    TensorStatus st = util_binary_op_general_NUD(a, b, out, add);
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
    TensorStatus st = util_ternary_op_general_NUD(a, b, c, out, add3);
    assert(st == TENSOR_OK);
    float expected[] = {1 + 10 + 100, 2 + 20 + 200, 3 + 10 + 100, 4 + 20 + 200};
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

/* ==================== 新增测试 ==================== */

static void test_broadcast_shapes()
{
    // 测试多个形状广播
    const int *dims[3];
    int ndims[3];
    int a_dims[] = {2, 1, 3};
    int b_dims[] = {1, 3, 1};
    int c_dims[] = {3, 1};
    dims[0] = a_dims;
    dims[1] = b_dims;
    dims[2] = c_dims;
    ndims[0] = 3;
    ndims[1] = 3;
    ndims[2] = 2;

    int out_dims[3];
    int out_ndim;
    int ok = util_broadcast_shapes(dims, ndims, 3, out_dims, &out_ndim);
    assert(ok == 1);
    assert(out_ndim == 3);
    assert(out_dims[0] == 2 && out_dims[1] == 3 && out_dims[2] == 3); // 逐维max: max(2,1,1)=2, max(1,3,3)=3, max(3,1,1)=3

    // 不兼容情况
    int d_dims[] = {2, 2};
    dims[2] = d_dims;
    ndims[2] = 2;
    ok = util_broadcast_shapes(dims, ndims, 3, out_dims, &out_ndim);
    assert(ok == 0); // 第二维 1 vs 2 不可广播

    // 测试空数组
    ok = util_broadcast_shapes(NULL, NULL, 0, out_dims, &out_ndim);
    assert(ok == 1);
    assert(out_ndim == 0);

    printf("test_broadcast_shapes passed\n");
}

static void test_shapes_equal()
{
    int a[] = {2, 3, 4};
    int b[] = {2, 3, 4};
    int c[] = {2, 3, 5};
    assert(util_shapes_equal(a, b, 3) == 1);
    assert(util_shapes_equal(a, c, 3) == 0);
    assert(util_shapes_equal(a, a, 3) == 1);
    // ndim=0 时，空数组视为相等？util_shapes_equal 循环 0 次，返回 1
    assert(util_shapes_equal(NULL, NULL, 0) == 1);
    printf("test_shapes_equal passed\n");
}

static void test_increment_coords()
{
    int dims[] = {2, 3, 2};
    int ndim = 3;
    int coords[3] = {0, 0, 0};
    // 遍历所有坐标，计数应与总元素数相等
    int count = 0;
    do
    {
        count++;
    } while (!util_increment_coords(coords, dims, ndim));
    assert(count == 2 * 3 * 2);

    // 测试边界：增量到最后一个
    coords[0] = 1;
    coords[1] = 2;
    coords[2] = 1;
    int done = util_increment_coords(coords, dims, ndim);
    assert(done == 1); // 已越界

    // 测试空维度 (ndim=0)
    done = util_increment_coords(NULL, NULL, 0);
    assert(done == 1); // 直接返回 1 表示已结束

    printf("test_increment_coords passed\n");
}

static void test_coords_from_linear()
{
    int dims[] = {2, 3, 4};
    int ndim = 3;
    int coords[3];
    util_coords_from_linear(0, dims, ndim, coords);
    assert(coords[0] == 0 && coords[1] == 0 && coords[2] == 0);
    util_coords_from_linear(5, dims, ndim, coords);
    assert(coords[0] == 0 && coords[1] == 1 && coords[2] == 1); // 5 = 0*12 + 1*4 + 1
    util_coords_from_linear(23, dims, ndim, coords);
    assert(coords[0] == 1 && coords[1] == 2 && coords[2] == 3); // 23 = 1*12 + 2*4 + 3
    // 测试 ndim=0
    util_coords_from_linear(0, NULL, 0, NULL); // 无操作，仅确保不崩溃
    printf("test_coords_from_linear passed\n");
}

static void test_same_data()
{
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    a->data[0] = 1;
    a->data[1] = 2;
    a->data[2] = 3;
    a->data[3] = 4;
    Tensor *b = tensor_view(a, 2, dims, NULL); // 共享数据
    Tensor *c = tensor_clone(a);               // 独立副本

    assert(util_same_data(a, a) == 1);
    assert(util_same_data(a, b) == 1);
    assert(util_same_data(a, c) == 0);
    assert(util_same_data(b, c) == 0);

    // 外部数据无引用计数，但数据指针相同，应视为共享
    float data[4] = {1, 2, 3, 4};
    Tensor *d = tensor_wrap(data, 2, dims, NULL);
    Tensor *e = tensor_wrap(data, 2, dims, NULL); // 不同包装，但数据指针相同
    assert(util_same_data(d, e) == 1);            // 指针相同，应视为共享

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(c);
    tensor_destroy(d);
    tensor_destroy(e);
    printf("test_same_data passed\n");
}

static void test_clear_tensor()
{
    int dims[] = {2, 3};
    Tensor *t = tensor_create(2, dims);
    for (size_t i = 0; i < t->size; ++i)
        t->data[i] = (float)i + 1.0f;
    util_clear_tensor(t);
    for (size_t i = 0; i < t->size; ++i)
        assert(t->data[i] == 0.0f);

    // 测试非连续张量
    int strides[] = {1, 2}; // 转置步长
    int new_dims[] = {3, 2};
    Tensor *v = tensor_view(t, 2, new_dims, strides);
    util_clear_tensor(v);
    // 通过 v 访问，所有元素应为 0
    for (size_t i = 0; i < v->size; ++i)
    {
        ptrdiff_t off = i * 1; // v 不连续，需用索引？但 v->data 是原始数据，清除后所有原数据为0
    }
    // 直接检查 t 所有元素为0
    for (size_t i = 0; i < t->size; ++i)
        assert(t->data[i] == 0.0f);

    tensor_destroy(t);
    tensor_destroy(v);
    printf("test_clear_tensor passed\n");
}

static void test_random_double_vector()
{
    double v[10];
    util_random_double_vector(v, 10);
    for (int i = 0; i < 10; ++i)
    {
        assert(v[i] >= -1.0 && v[i] <= 1.0);
    }
    // 测试 len=0
    util_random_double_vector(NULL, 0); // 应无操作
    printf("test_random_double_vector passed\n");
}

static void test_copy_ints()
{
    int src[] = {1, 2, 3, 4};
    int *dst = util_copy_ints(src, 4);
    assert(dst != NULL);
    assert(memcmp(src, dst, 4 * sizeof(int)) == 0);
    free(dst);
    dst = util_copy_ints(src, 0);
    assert(dst == NULL);
    printf("test_copy_ints passed\n");
}

static void test_calc_contiguous_strides()
{
    int dims[] = {2, 3, 4};
    int *strides = util_calc_contiguous_strides(dims, 3);
    assert(strides[0] == 12 && strides[1] == 4 && strides[2] == 1);
    free(strides);

    strides = util_calc_contiguous_strides(dims, 0);
    assert(strides == NULL);
    printf("test_calc_contiguous_strides passed\n");
}

static void test_calc_size()
{
    int dims[] = {2, 3, 4};
    size_t sz = util_calc_size(dims, 3);
    assert(sz == 24);
    sz = util_calc_size(NULL, 0);
    assert(sz == 1);
    printf("test_calc_size passed\n");
}

static void test_float_to_index()
{
    TensorStatus st;
    int idx = tensor_float_to_index(3.0f, 5, &st);
    assert(st == TENSOR_OK && idx == 3);
    idx = tensor_float_to_index(-1.0f, 5, &st);
    assert(st == TENSOR_OK && idx == 4);
    idx = tensor_float_to_index(-6.0f, 5, &st);
    assert(st == TENSOR_ERR_INDEX_OUT_OF_BOUNDS);
    idx = tensor_float_to_index(5.0f, 5, &st);
    assert(st == TENSOR_ERR_INDEX_OUT_OF_BOUNDS);
    idx = tensor_float_to_index(NAN, 5, &st);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    idx = tensor_float_to_index(2.5f, 5, &st);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    printf("test_float_to_index passed\n");
}

/* ==================== 主函数 ==================== */

int main()
{
    // 原有测试
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

    // 新增测试
    test_broadcast_shapes();
    test_shapes_equal();
    test_increment_coords();
    test_coords_from_linear();
    test_same_data();
    test_clear_tensor();
    test_random_double_vector();
    test_copy_ints();
    test_calc_contiguous_strides();
    test_calc_size();
    test_float_to_index();

    printf("All utils tests passed!\n");
    return 0;
}