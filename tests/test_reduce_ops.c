#include "tensor.h"
#include "reduce_ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")
#define EPS 1e-5f

/* 辅助：比较两个浮点数是否近似相等 */
static int approx_equal(float a, float b, float eps)
{
    if (isnan(a) && isnan(b))
        return 1;
    if (isinf(a) && isinf(b))
        return (a > 0) == (b > 0);
    return fabsf(a - b) < eps;
}

/* 辅助：检查张量所有元素与预期值是否相等 */
static int check_tensor(const Tensor *t, const float *expected, size_t n)
{
    if (tensor_size(t) != n)
        return 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (!approx_equal(t->data[i], expected[i], EPS))
            return 0;
    }
    return 1;
}

/* 辅助：打印张量（调试用） */
static void print_tensor(const Tensor *t, const char *name)
{
    printf("%s: ", name);
    for (size_t i = 0; i < t->size; ++i)
        printf("%.2f ", t->data[i]);
    printf("\n");
}

/* 测试求和 */
void test_sum()
{
    TEST("tensor_sum");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    // 沿轴0归约，keepdims=0
    int out_dims0[] = {3};
    Tensor *out0 = tensor_create(1, out_dims0);
    tensor_sum(x, out0, 0, 0);
    float expected0[] = {5, 7, 9}; // 1+4, 2+5, 3+6
    assert(check_tensor(out0, expected0, 3));

    // 沿轴0归约，keepdims=1
    int out_dims0k[] = {1, 3};
    Tensor *out0k = tensor_create(2, out_dims0k);
    tensor_sum(x, out0k, 0, 1);
    float expected0k[] = {5, 7, 9};
    assert(check_tensor(out0k, expected0k, 3));

    // 沿轴1归约，keepdims=0
    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_sum(x, out1, 1, 0);
    float expected1[] = {6, 15}; // 1+2+3, 4+5+6
    assert(check_tensor(out1, expected1, 2));

    // 沿轴1归约，keepdims=1
    int out_dims1k[] = {2, 1};
    Tensor *out1k = tensor_create(2, out_dims1k);
    tensor_sum(x, out1k, 1, 1);
    float expected1k[] = {6, 15};
    assert(check_tensor(out1k, expected1k, 2));

    // 归约所有轴 (axis=-1), keepdims=0 -> 标量 (0维)
    Tensor *out_all = tensor_create(0, NULL);
    tensor_sum(x, out_all, -1, 0);
    assert(out_all->size == 1);
    assert(approx_equal(out_all->data[0], 21.0f, EPS));

    // 归约所有轴, keepdims=1 -> 1x1x1? 实际上是2维保持，但所有轴大小为1
    int out_dims_allk[] = {1, 1};
    Tensor *out_allk = tensor_create(2, out_dims_allk);
    tensor_sum(x, out_allk, -1, 1);
    float expected_allk[] = {21};
    assert(check_tensor(out_allk, expected_allk, 1));

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out0k);
    tensor_destroy(out1);
    tensor_destroy(out1k);
    tensor_destroy(out_all);
    tensor_destroy(out_allk);
    PASS();
}

/* 测试乘积 */
void test_prod()
{
    TEST("tensor_prod");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    // 沿轴0
    int out_dims0[] = {3};
    Tensor *out0 = tensor_create(1, out_dims0);
    tensor_prod(x, out0, 0, 0);
    float expected0[] = {4, 10, 18}; // 1*4, 2*5, 3*6
    assert(check_tensor(out0, expected0, 3));

    // 沿轴1
    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_prod(x, out1, 1, 0);
    float expected1[] = {6, 120}; // 1*2*3, 4*5*6
    assert(check_tensor(out1, expected1, 2));

    // 所有轴
    Tensor *out_all = tensor_create(0, NULL);
    tensor_prod(x, out_all, -1, 0);
    assert(approx_equal(out_all->data[0], 720.0f, EPS));

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out1);
    tensor_destroy(out_all);
    PASS();
}

/* 测试最大值 */
void test_max()
{
    TEST("tensor_max");
    int dims[] = {2, 3};
    float data[] = {1, 5, 3, 4, 2, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    // 沿轴0
    int out_dims0[] = {3};
    Tensor *out0 = tensor_create(1, out_dims0);
    tensor_max(x, out0, 0, 0);
    float expected0[] = {4, 5, 6};
    assert(check_tensor(out0, expected0, 3));

    // 沿轴1
    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_max(x, out1, 1, 0);
    float expected1[] = {5, 6};
    assert(check_tensor(out1, expected1, 2));

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out1);
    PASS();
}

/* 测试最小值 */
void test_min()
{
    TEST("tensor_min");
    int dims[] = {2, 3};
    float data[] = {1, 5, 3, 4, 2, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_min(x, out1, 1, 0);
    float expected1[] = {1, 2};
    assert(check_tensor(out1, expected1, 2));

    tensor_destroy(x);
    tensor_destroy(out1);
    PASS();
}

/* 测试 argmax */
void test_argmax()
{
    TEST("tensor_argmax");
    int dims[] = {2, 3};
    float data[] = {1, 5, 3, 4, 2, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    // 沿轴0 (比较每一列)
    int out_dims0[] = {3};
    Tensor *out0 = tensor_create(1, out_dims0);
    tensor_argmax(x, out0, 0, 0);
    float expected0[] = {1, 0, 1}; // 第0列最大4在行1，第1列最大5在行0，第2列最大6在行1
    assert(check_tensor(out0, expected0, 3));

    // 沿轴1 (比较每一行)
    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_argmax(x, out1, 1, 0);
    float expected1[] = {1, 2}; // 第0行最大5在索引1，第1行最大6在索引2
    assert(check_tensor(out1, expected1, 2));

    // 所有轴，返回全局最大值的线性索引（扁平化索引）
    Tensor *out_all = tensor_create(0, NULL);
    tensor_argmax(x, out_all, -1, 0);
    assert(approx_equal(out_all->data[0], 5.0f, EPS)); // 6 在位置5

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out1);
    tensor_destroy(out_all);
    PASS();
}

/* 测试 argmin */
void test_argmin()
{
    TEST("tensor_argmin");
    int dims[] = {2, 3};
    float data[] = {1, 5, 3, 4, 2, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_argmin(x, out1, 1, 0);
    float expected1[] = {0, 1}; // 第0行最小1在索引0，第1行最小2在索引1
    assert(check_tensor(out1, expected1, 2));

    tensor_destroy(x);
    tensor_destroy(out1);
    PASS();
}

/* 测试 any */
void test_any()
{
    TEST("tensor_any");
    int dims[] = {2, 3};
    float data[] = {0, 0, 0, 0, 0, 1};
    Tensor *x = tensor_from_array(data, 2, dims);

    // 沿轴0
    int out_dims0[] = {3};
    Tensor *out0 = tensor_create(1, out_dims0);
    tensor_any(x, out0, 0, 0);
    float expected0[] = {0, 0, 1}; // 每列是否有非零
    assert(check_tensor(out0, expected0, 3));

    // 沿轴1
    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_any(x, out1, 1, 0);
    float expected1[] = {0, 1};
    assert(check_tensor(out1, expected1, 2));

    // 所有轴
    Tensor *out_all = tensor_create(0, NULL);
    tensor_any(x, out_all, -1, 0);
    assert(approx_equal(out_all->data[0], 1.0f, EPS));

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out1);
    tensor_destroy(out_all);
    PASS();
}

/* 测试 all */
void test_all()
{
    TEST("tensor_all");
    int dims[] = {2, 3};
    float data[] = {1, 1, 1, 1, 1, 0};
    Tensor *x = tensor_from_array(data, 2, dims);

    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_all(x, out1, 1, 0);
    float expected1[] = {1, 0}; // 第0行全非零，第1行有0
    assert(check_tensor(out1, expected1, 2));

    tensor_destroy(x);
    tensor_destroy(out1);
    PASS();
}

/* 测试均值 */
void test_mean()
{
    TEST("tensor_mean");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_mean(x, out1, 1, 0);
    float expected1[] = {(1 + 2 + 3) / 3.0f, (4 + 5 + 6) / 3.0f};
    assert(check_tensor(out1, expected1, 2));

    tensor_destroy(x);
    tensor_destroy(out1);
    PASS();
}

/* 测试方差 */
void test_var()
{
    TEST("tensor_var");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *x = tensor_from_array(data, 1, dims);

    // 总体方差 (unbiased=0)
    Tensor *out0 = tensor_create(0, NULL);
    tensor_var(x, out0, -1, 0, 0);
    float var_pop = ((1 - 2) * (1 - 2) + (2 - 2) * (2 - 2) + (3 - 2) * (3 - 2)) / 3.0f; // 2/3≈0.6667
    assert(approx_equal(out0->data[0], var_pop, EPS));

    // 样本方差 (unbiased=1)
    Tensor *out1 = tensor_create(0, NULL);
    tensor_var(x, out1, -1, 0, 1);
    float var_sample = ((1 - 2) * (1 - 2) + (2 - 2) * (2 - 2) + (3 - 2) * (3 - 2)) / 2.0f; // 1.0
    assert(approx_equal(out1->data[0], var_sample, EPS));

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out1);
    PASS();
}

/* 测试标准差 */
void test_std()
{
    TEST("tensor_std");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *x = tensor_from_array(data, 1, dims);

    Tensor *out = tensor_create(0, NULL);
    tensor_std(x, out, -1, 0, 0);
    float std_pop = sqrtf(((1 - 2) * (1 - 2) + (2 - 2) * (2 - 2) + (3 - 2) * (3 - 2)) / 3.0f);
    assert(approx_equal(out->data[0], std_pop, EPS));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

/* 测试范数 */
void test_norm()
{
    TEST("tensor_norm");
    int dims[] = {3};
    float data[] = {-1, 2, -3};
    Tensor *x = tensor_from_array(data, 1, dims);

    // L1 范数
    Tensor *out1 = tensor_create(0, NULL);
    tensor_norm(x, out1, -1, 0, 1.0f);
    assert(approx_equal(out1->data[0], 1 + 2 + 3, EPS));

    // L2 范数
    Tensor *out2 = tensor_create(0, NULL);
    tensor_norm(x, out2, -1, 0, 2.0f);
    float l2 = sqrtf(1 + 4 + 9); // sqrt(14)≈3.7417
    assert(approx_equal(out2->data[0], l2, EPS));

    // L0 范数（非零个数）
    Tensor *out0 = tensor_create(0, NULL);
    tensor_norm(x, out0, -1, 0, 0.0f);
    assert(approx_equal(out0->data[0], 3.0f, EPS));

    tensor_destroy(x);
    tensor_destroy(out1);
    tensor_destroy(out2);
    tensor_destroy(out0);
    PASS();
}

/* 测试不连续张量（视图）的归约 */
void test_noncontiguous()
{
    TEST("tensor_reduce noncontiguous");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *a = tensor_from_array(data, 2, dims);

    // 创建转置视图 (3,2)
    int new_dims[] = {3, 2};
    int strides[] = {1, 3};
    Tensor *v = tensor_view(a, 2, new_dims, strides);
    assert(v != NULL);

    // 对视图沿轴0求和，预期结果：每一列的和（原矩阵的行和？需要计算）
    // 原数据 a: [1 2 3; 4 5 6]
    // 转置后 v: [1 4; 2 5; 3 6]  (行主序视图，但步长使得存储顺序不同)
    // 沿轴0归约（keepdims=0）应得到每列的和： [1+2+3, 4+5+6] = [6, 15]
    int out_dims[] = {2};
    Tensor *out = tensor_create(1, out_dims);
    tensor_sum(v, out, 0, 0);
    float expected[] = {6, 15};
    assert(check_tensor(out, expected, 2));

    tensor_destroy(a);
    tensor_destroy(v);
    tensor_destroy(out);
    PASS();
}

int main()
{
    test_sum();
    test_prod();
    test_max();
    test_min();
    test_argmax();
    test_argmin();
    test_any();
    test_all();
    test_mean();
    test_var();
    test_std();
    test_norm();
    test_noncontiguous();

    printf("All reduce_ops tests passed!\n");
    return 0;
}