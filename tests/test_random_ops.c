#include "tensor.h"
#include "random_ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")
#define EPS 1e-2f // 随机统计允许较大误差
#define N 10000   // 大样本数

/* 辅助：检查张量是否在给定范围内 */
static int check_range(const Tensor *t, float low, float high)
{
    for (size_t i = 0; i < t->size; ++i)
    {
        float v = t->data[i];
        if (v < low || v >= high)
            return 0;
    }
    return 1;
}

/* 辅助：检查整数张量是否在范围内且为整数 */
static int check_int_range(const Tensor *t, int low, int high)
{
    for (size_t i = 0; i < t->size; ++i)
    {
        int v = (int)t->data[i];
        if (v < low || v >= high)
            return 0;
        if (fabsf(t->data[i] - v) > 1e-6f) // 确保是整数
            return 0;
    }
    return 1;
}

/* 辅助：计算均值（标量张量） */
static float mean_scalar(const Tensor *t)
{
    double sum = 0.0;
    for (size_t i = 0; i < t->size; ++i)
        sum += t->data[i];
    return (float)(sum / t->size);
}

/* 辅助：计算标准差（标量张量，总体方差） */
static float std_scalar(const Tensor *t)
{
    double mean = mean_scalar(t);
    double sum = 0.0;
    for (size_t i = 0; i < t->size; ++i)
    {
        double d = t->data[i] - mean;
        sum += d * d;
    }
    return (float)sqrt(sum / t->size);
}

/* ---------- 测试种子 ---------- */
void test_seed()
{
    TEST("tensor_random_seed");
    tensor_random_seed(12345);
    int r1 = rand();
    tensor_random_seed(12345);
    int r2 = rand();
    assert(r1 == r2);
    PASS();
}

/* ---------- 测试均匀分布 ---------- */
void test_uniform()
{
    TEST("tensor_random_uniform");
    int dims[] = {1000};
    Tensor *t = tensor_create(1, dims);
    tensor_random_uniform(t, -1.0f, 2.0f);
    assert(check_range(t, -1.0f, 2.0f));
    float m = mean_scalar(t);
    float s = std_scalar(t);
    // 理论均值 0.5，方差 0.75，标准差≈0.866
    assert(fabsf(m - 0.5f) < 0.1f);
    assert(fabsf(s - 0.866f) < 0.1f);
    tensor_destroy(t);
    PASS();
}

/* ---------- 测试正态分布 ---------- */
void test_normal()
{
    TEST("tensor_random_normal");
    int dims[] = {5000};
    Tensor *t = tensor_create(1, dims);
    tensor_random_normal(t, 3.0f, 2.0f);
    float m = mean_scalar(t);
    float s = std_scalar(t);
    assert(fabsf(m - 3.0f) < 0.2f);
    assert(fabsf(s - 2.0f) < 0.2f);
    tensor_destroy(t);
    PASS();
}

/* ---------- 测试截断正态分布 ---------- */
void test_truncated_normal()
{
    TEST("tensor_random_truncated_normal");
    int dims[] = {5000};
    Tensor *t = tensor_create(1, dims);
    tensor_random_truncated_normal(t, 0.0f, 1.0f, -1.0f, 1.0f);
    assert(check_range(t, -1.0f, 1.0f));
    float m = mean_scalar(t);
    float s = std_scalar(t);
    assert(fabsf(m) < 0.1f);
    assert(s < 1.0f && s > 0.5f);
    tensor_destroy(t);
    PASS();
}

/* ---------- 测试伯努利分布 ---------- */
void test_bernoulli()
{
    TEST("tensor_random_bernoulli");
    int dims[] = {5000};
    Tensor *t = tensor_create(1, dims);
    tensor_random_bernoulli(t, 0.3f);
    assert(check_range(t, 0.0f, 1.1f)); // 只检查0或1
    float m = mean_scalar(t);
    assert(fabsf(m - 0.3f) < 0.05f);
    tensor_destroy(t);
    PASS();
}

/* ---------- 测试随机整数 ---------- */
void test_randint()
{
    TEST("tensor_random_randint");
    int dims[] = {1000};
    Tensor *t = tensor_create(1, dims);
    tensor_random_randint(t, 5, 10);
    assert(check_int_range(t, 5, 10));
    float m = mean_scalar(t);
    assert(fabsf(m - 7.0f) < 0.5f);
    tensor_destroy(t);
    PASS();
}

/* ---------- 测试打乱 ---------- */
void test_shuffle()
{
    TEST("tensor_shuffle");
    int dims[] = {5, 3};
    float data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *dst = tensor_create(2, dims);

    tensor_random_seed(42);
    tensor_shuffle(src, dst);

    // 检查总元素集合相同（排序后比较）
    float sorted_src[15], sorted_dst[15];
    memcpy(sorted_src, src->data, 15 * sizeof(float));
    memcpy(sorted_dst, dst->data, 15 * sizeof(float));
    // 简单排序（冒泡）
    for (int i = 0; i < 15; ++i)
        for (int j = i + 1; j < 15; ++j)
        {
            if (sorted_src[i] > sorted_src[j])
            {
                float t = sorted_src[i];
                sorted_src[i] = sorted_src[j];
                sorted_src[j] = t;
            }
            if (sorted_dst[i] > sorted_dst[j])
            {
                float t = sorted_dst[i];
                sorted_dst[i] = sorted_dst[j];
                sorted_dst[j] = t;
            }
        }
    for (int i = 0; i < 15; ++i)
        assert(sorted_src[i] == sorted_dst[i]);

    // 检查第一维被打乱了（至少不是完全相同的顺序）
    int same_count = 0;
    for (int i = 0; i < 5; ++i)
    {
        int equal = 1;
        for (int j = 0; j < 3; ++j)
            if (src->data[i * 3 + j] != dst->data[i * 3 + j])
                equal = 0;
        if (equal)
            same_count++;
    }
    assert(same_count < 5); // 至少有一行不同

    tensor_destroy(src);
    tensor_destroy(dst);
    PASS();
}

/* ---------- 错误测试：均匀分布参数无效 ---------- */
void test_uniform_error()
{
    TEST("tensor_random_uniform error (low >= high)");
    int dims[] = {10};
    Tensor *t = tensor_create(1, dims);
    TensorStatus st = tensor_random_uniform(t, 5.0f, 5.0f); // low == high
    assert(st == TENSOR_ERR_INVALID_PARAM);
    st = tensor_random_uniform(t, 5.0f, 3.0f); // low > high
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(t);
    PASS();
}

/* ---------- 错误测试：正态分布参数无效 ---------- */
void test_normal_error()
{
    TEST("tensor_random_normal error (std < 0)");
    int dims[] = {10};
    Tensor *t = tensor_create(1, dims);
    TensorStatus st = tensor_random_normal(t, 0.0f, -1.0f);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(t);
    PASS();
}

/* ---------- 错误测试：截断正态分布参数无效 ---------- */
void test_truncated_normal_error()
{
    TEST("tensor_random_truncated_normal error");
    int dims[] = {10};
    Tensor *t = tensor_create(1, dims);

    // std < 0
    TensorStatus st = tensor_random_truncated_normal(t, 0.0f, -1.0f, -1.0f, 1.0f);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    // a >= b
    st = tensor_random_truncated_normal(t, 0.0f, 1.0f, 1.0f, 1.0f); // a==b
    assert(st == TENSOR_ERR_INVALID_PARAM);
    st = tensor_random_truncated_normal(t, 0.0f, 1.0f, 2.0f, 1.0f); // a>b
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(t);
    PASS();
}

/* ---------- 错误测试：伯努利分布参数无效 ---------- */
void test_bernoulli_error()
{
    TEST("tensor_random_bernoulli error (p out of [0,1])");
    int dims[] = {10};
    Tensor *t = tensor_create(1, dims);

    TensorStatus st = tensor_random_bernoulli(t, -0.1f);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    st = tensor_random_bernoulli(t, 1.1f);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(t);
    PASS();
}

/* ---------- 错误测试：随机整数参数无效 ---------- */
void test_randint_error()
{
    TEST("tensor_random_randint error (low >= high)");
    int dims[] = {10};
    Tensor *t = tensor_create(1, dims);

    TensorStatus st = tensor_random_randint(t, 5, 5); // low == high
    assert(st == TENSOR_ERR_INVALID_PARAM);

    st = tensor_random_randint(t, 10, 5); // low > high
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(t);
    PASS();
}

/* ---------- 错误测试：shuffle 形状不匹配 ---------- */
void test_shuffle_shape_mismatch()
{
    TEST("tensor_shuffle shape mismatch");
    int dims_src[] = {5, 3};
    int dims_dst[] = {5, 4}; // 不同
    Tensor *src = tensor_create(2, dims_src);
    Tensor *dst = tensor_create(2, dims_dst);

    TensorStatus st = tensor_shuffle(src, dst);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    tensor_destroy(src);
    tensor_destroy(dst);
    PASS();
}

/* ---------- 错误测试：shuffle 输入维度小于1 ---------- */
void test_shuffle_ndim_error()
{
    TEST("tensor_shuffle ndim < 1");
    Tensor *src = tensor_create(0, NULL); // 标量，ndim=0
    int dims_dst[] = {1};
    Tensor *dst = tensor_create(1, dims_dst);

    TensorStatus st = tensor_shuffle(src, dst);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(src);
    tensor_destroy(dst);
    PASS();
}

/* ---------- 主函数 ---------- */
int main()
{
    test_seed();
    test_uniform();
    test_normal();
    test_truncated_normal();
    test_bernoulli();
    test_randint();
    test_shuffle();

    test_uniform_error();
    test_normal_error();
    test_truncated_normal_error();
    test_bernoulli_error();
    test_randint_error();
    test_shuffle_shape_mismatch();
    test_shuffle_ndim_error();

    printf("All random_ops tests passed!\n");
    return 0;
}