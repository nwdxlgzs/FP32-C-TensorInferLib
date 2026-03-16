#include "tensor.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

/* 辅助：打印测试结果 */
#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")

void test_create_destroy()
{
    TEST("test_create_destroy");
    int dims[] = {2, 3};
    Tensor *t = tensor_create(2, dims);
    assert(t != NULL);
    assert(tensor_ndim(t) == 2);
    assert(tensor_dims(t)[0] == 2 && tensor_dims(t)[1] == 3);
    assert(tensor_size(t) == 6);
    assert(tensor_strides(t) != NULL);
    assert(tensor_strides(t)[0] == 3 && tensor_strides(t)[1] == 1);
    tensor_destroy(t);
    PASS();
}

void test_create_scalar()
{
    TEST("test_create_scalar");
    Tensor *t = tensor_create(0, NULL);
    assert(t != NULL);
    assert(tensor_ndim(t) == 0);
    assert(tensor_dims(t) == NULL);
    assert(tensor_strides(t) == NULL);
    assert(tensor_size(t) == 1);
    tensor_destroy(t);
    PASS();
}

void test_wrap()
{
    TEST("test_wrap");
    float data[6] = {1, 2, 3, 4, 5, 6};
    int dims[] = {2, 3};
    Tensor *t = tensor_wrap(data, 2, dims, NULL);
    assert(t != NULL);
    assert(tensor_ndim(t) == 2);
    assert(tensor_size(t) == 6);
    assert(tensor_strides(t) == NULL); // 标记连续
    size_t off = tensor_offset(t, (int[]){1, 2});
    assert(off == 1 * 3 + 2);
    assert(t->data[off] == 6);
    tensor_destroy(t); // 不会释放data
    PASS();
}

void test_from_array()
{
    TEST("test_from_array");
    float data[6] = {1, 2, 3, 4, 5, 6};
    int dims[] = {2, 3};
    Tensor *t = tensor_from_array(data, 2, dims);
    assert(t != NULL);
    assert(t->data[0] == 1);
    data[0] = 100; // 修改原数组不影响t
    assert(t->data[0] == 1);
    tensor_destroy(t);
    PASS();
}

void test_clone()
{
    TEST("test_clone");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 4; ++i)
        a->data[i] = i;
    Tensor *b = tensor_clone(a);
    assert(b != NULL);
    assert(b->data[0] == 0 && b->data[3] == 3);
    a->data[0] = 100;
    assert(b->data[0] == 0);
    tensor_destroy(a);
    tensor_destroy(b);
    PASS();
}

void test_view_reshape()
{
    TEST("test_view_reshape");
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    // 创建视图 (3,2) 通过 reshape（原连续，所以自动计算连续步长）
    int new_dims[] = {3, 2};
    Tensor *v = tensor_view(a, 2, new_dims, NULL);
    assert(v != NULL);
    assert(v->strides[0] == 2 && v->strides[1] == 1); // 自动计算的连续步长
    // 验证数据对应关系
    assert(v->data[0] == a->data[0]); // v(0,0) -> a(0,0)
    int idx_v[2] = {2, 1};            // 对应 a 的哪个？a 连续步长(3,1)，v(2,1) 线性索引 2*2+1=5，对应 a 的线性索引 5，即 a(1,2) 值为 5
    size_t off_v = tensor_offset(v, idx_v);
    assert(off_v == 5);
    assert(v->data[off_v] == 5);
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_view_with_strides()
{
    TEST("test_view_with_strides");
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    // 创建转置视图：形状 (3,2)，步长 (1,3) 使得 v(i,j) = a(j,i)
    int new_dims[] = {3, 2};
    int strides[] = {1, 3};
    Tensor *v = tensor_view(a, 2, new_dims, strides);
    assert(v != NULL);
    assert(v->strides[0] == 1 && v->strides[1] == 3);
    // 验证转置：
    // v(0,0) = a(0,0) = 0
    // v(0,1) = a(1,0) = 3
    // v(1,0) = a(0,1) = 1
    // v(2,1) = a(1,2) = 5
    assert(v->data[tensor_offset(v, (int[]){0, 0})] == 0);
    assert(v->data[tensor_offset(v, (int[]){0, 1})] == 3);
    assert(v->data[tensor_offset(v, (int[]){1, 0})] == 1);
    assert(v->data[tensor_offset(v, (int[]){2, 1})] == 5);
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_view_noncontiguous_fail()
{
    TEST("test_view_noncontiguous_fail");
    // 创建一个不连续的张量（通过视图转置）
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    int new_dims[] = {3, 2};
    int strides[] = {1, 3};
    Tensor *v = tensor_view(a, 2, new_dims, strides); // v 不连续
    assert(v != NULL);
    // 尝试在不连续的原张量上创建视图而不提供步长，应该失败
    int new_dims2[] = {1, 6};
    Tensor *w = tensor_view(v, 2, new_dims2, NULL);
    assert(w == NULL); // 应该失败
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_copy()
{
    TEST("test_copy");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 4; ++i)
        a->data[i] = i;
    Tensor *b = tensor_create(2, dims);
    TensorStatus status = tensor_copy(b, a);
    assert(status == TENSOR_OK);
    assert(b->data[0] == 0 && b->data[3] == 3);
    a->data[0] = 100;
    assert(b->data[0] == 0);
    tensor_destroy(a);
    tensor_destroy(b);
    PASS();
}

void test_contiguous()
{
    TEST("test_contiguous");
    int dims[] = {3, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    // 创建转置视图 v: 形状 (2,3)，步长 (1,2) 使得 v(i,j) = a(j,i)
    int new_dims[] = {2, 3};
    int strides[] = {1, 2}; // 修正：正确的转置步长
    Tensor *v = tensor_view(a, 2, new_dims, strides);
    assert(!util_is_contiguous(v)); // 不连续
    // 连续化
    TensorStatus status = tensor_contiguous(v);
    assert(status == TENSOR_OK);
    assert(util_is_contiguous(v));
    // 验证数据：转置后的连续数据应为 [0,2,4,1,3,5]
    float expected[] = {0, 2, 4, 1, 3, 5};
    for (int i = 0; i < 6; ++i)
        assert(v->data[i] == expected[i]);
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_contiguous_external_fail()
{
    TEST("test_contiguous_external_fail");
    float data[6] = {0, 1, 2, 3, 4, 5};
    int dims[] = {3, 2};
    // 包装为连续张量
    Tensor *t = tensor_wrap(data, 2, dims, NULL);
    // 制造一个不连续的外部视图？但 wrap 只能包装连续数据，除非提供非连续步长
    // 这里测试外部数据且不连续的情况：提供非连续步长
    int strides[] = {1, 3};
    Tensor *v = tensor_wrap(data, 2, dims, strides); // 外部数据，不连续
    assert(v != NULL);
    assert(!util_is_contiguous(v));
    // 连续化应该失败，因为外部数据无法原地变连续
    TensorStatus status = tensor_contiguous(v);
    assert(status == TENSOR_ERR_UNSUPPORTED);
    tensor_destroy(v);
    // 用 wrap 连续数据则没问题
    Tensor *w = tensor_wrap(data, 2, dims, NULL);
    status = tensor_contiguous(w);
    assert(status == TENSOR_OK); // 已经是连续
    tensor_destroy(w);
    PASS();
}

void test_offset()
{
    TEST("test_offset");
    int dims[] = {2, 3, 4};
    Tensor *t = tensor_create(3, dims);
    size_t off = tensor_offset(t, (int[]){1, 2, 3});
    assert(off == 1 * 12 + 2 * 4 + 3); // 12+8+3=23
    // 负索引
    off = tensor_offset(t, (int[]){-1, -1, -1});
    assert(off == 1 * 12 + 2 * 4 + 3);
    // 越界
    off = tensor_offset(t, (int[]){2, 0, 0});
    assert(off == SIZE_MAX);
    tensor_destroy(t);
    PASS();
}

void test_make_unique()
{
    TEST("test_make_unique");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 4; ++i)
        a->data[i] = i;
    Tensor *b = tensor_view(a, 2, dims, NULL); // 共享数据
    assert(*(a->ref_count) == 2);
    // 通过 b 触发写时拷贝（模拟修改）
    TensorStatus status = tensor_make_unique(b);
    assert(status == TENSOR_OK);
    assert(*(a->ref_count) == 1);
    assert(*(b->ref_count) == 1);
    assert(a->data != b->data);
    // 修改 b 不影响 a
    b->data[0] = 100;
    assert(a->data[0] == 0);
    tensor_destroy(a);
    tensor_destroy(b);
    PASS();
}

int main()
{
    test_create_destroy();
    test_create_scalar();
    test_wrap();
    test_from_array();
    test_clone();
    test_view_reshape();
    test_view_with_strides();
    test_view_noncontiguous_fail();
    test_copy();
    test_contiguous();
    test_contiguous_external_fail();
    test_offset();
    test_make_unique();
    printf("All tensor tests passed!\n");
    return 0;
}