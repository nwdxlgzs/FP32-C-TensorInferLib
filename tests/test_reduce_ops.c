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

/* ==================== 原有测试函数 ==================== */

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

void test_median()
{
    TEST("tensor_median");
    int dims[] = {2, 3};
    float data[] = {1, 5, 3, 4, 2, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    int out_dims0[] = {3};
    Tensor *out0 = tensor_create(1, out_dims0);
    tensor_median(x, out0, 0, 0);
    float expected0[] = {2.5f, 3.5f, 4.5f}; // 每列中位数
    assert(check_tensor(out0, expected0, 3));

    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_median(x, out1, 1, 0);
    float expected1[] = {3, 4}; // 每行中位数
    assert(check_tensor(out1, expected1, 2));

    Tensor *out_all = tensor_create(0, NULL);
    tensor_median(x, out_all, -1, 0);
    assert(approx_equal(out_all->data[0], 3.5f, EPS)); // 全局中位数

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out1);
    tensor_destroy(out_all);
    PASS();
}

void test_mode()
{
    TEST("tensor_mode");
    int dims[] = {2, 4};
    float data[] = {1, 2, 2, 3, 2, 2, 1, 1};
    Tensor *x = tensor_from_array(data, 2, dims);

    int out_dims0[] = {4};
    Tensor *out0 = tensor_create(1, out_dims0);
    tensor_mode(x, out0, 0, 0);
    float expected0[] = {1, 2, 1, 1};
    assert(check_tensor(out0, expected0, 4));

    int out_dims1[] = {2};
    Tensor *out1 = tensor_create(1, out_dims1);
    tensor_mode(x, out1, 1, 0);
    float expected1[] = {2, 1}; // 每行众数
    assert(check_tensor(out1, expected1, 2));

    Tensor *out_all = tensor_create(0, NULL);
    tensor_mode(x, out_all, -1, 0);
    assert(approx_equal(out_all->data[0], 2.0f, EPS)); // 全局众数

    tensor_destroy(x);
    tensor_destroy(out0);
    tensor_destroy(out1);
    tensor_destroy(out_all);
    PASS();
}

void test_quantile()
{
    TEST("tensor_quantile");

    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *q = tensor_create(0, NULL); // 标量分位数

    int out_dims[] = {2};
    Tensor *out;

    // ----- 测试 q=0.5 (偶数个元素) -----
    q->data[0] = 0.5f;

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_LINEAR, out);
    float expected_linear[] = {2, 5}; // 线性插值
    assert(check_tensor(out, expected_linear, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_LOWER, out);
    float expected_lower[] = {2, 5}; // 较低值，索引1.0 -> lo=1
    assert(check_tensor(out, expected_lower, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_HIGHER, out);
    float expected_higher[] = {2, 5}; // 较高值，索引1.0 -> hi=1
    assert(check_tensor(out, expected_higher, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_MIDPOINT, out);
    float expected_midpoint[] = {2, 5}; // 中点，同线性但权重0.5时相同
    assert(check_tensor(out, expected_midpoint, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_NEAREST, out);
    float expected_nearest[] = {2, 5}; // 最近邻，权重0.5->hi=1
    assert(check_tensor(out, expected_nearest, 2));
    tensor_destroy(out);

    // ----- 测试 q=0.25 (需要插值) -----
    q->data[0] = 0.25f;

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_LINEAR, out);
    float expected_linear_25[] = {1.5f, 4.5f}; // (1+2)/2=1.5, (4+5)/2=4.5
    assert(check_tensor(out, expected_linear_25, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_LOWER, out);
    float expected_lower_25[] = {1, 4}; // 索引0.5 -> lo=0
    assert(check_tensor(out, expected_lower_25, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_HIGHER, out);
    float expected_higher_25[] = {2, 5}; // 索引0.5 -> hi=1
    assert(check_tensor(out, expected_higher_25, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_MIDPOINT, out);
    float expected_midpoint_25[] = {1.5f, 4.5f}; // (1+2)/2=1.5, (4+5)/2=4.5
    assert(check_tensor(out, expected_midpoint_25, 2));
    tensor_destroy(out);

    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_NEAREST, out);
    float expected_nearest_25[] = {2, 5}; // 权重0.5 -> hi=1
    assert(check_tensor(out, expected_nearest_25, 2));
    tensor_destroy(out);

    // ----- 测试 q=0 和 q=1 (边界) -----
    q->data[0] = 0.0f;
    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_LINEAR, out);
    float expected_0[] = {1, 4}; // 最小值
    assert(check_tensor(out, expected_0, 2));
    tensor_destroy(out);

    q->data[0] = 1.0f;
    out = tensor_create(1, out_dims);
    tensor_quantile(x, q, 1, 0, QUANTILE_LINEAR, out);
    float expected_1[] = {3, 6}; // 最大值
    assert(check_tensor(out, expected_1, 2));
    tensor_destroy(out);

    tensor_destroy(x);
    tensor_destroy(q);
    PASS();
}

/* ==================== 新增错误测试 ==================== */

void test_reduce_errors_axis()
{
    TEST("tensor_reduce errors: invalid axis");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(0, NULL); // 标量

    // axis 超出范围
    TensorStatus st = tensor_sum(x, out, 2, 0);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    st = tensor_sum(x, out, -3, 0); // -3 对于 2 维，归一化后为 -1? 但 -3 会超出范围
    // 根据 util_normalize_axis，-3 对于 2 维会变成 -1? 因为 -3+2=-1，-1 是合法轴。但 -1 表示所有轴，合法。因此我们需要测试一个真正越界的轴。
    // 测试 axis = 2 已经越界。测试 axis = -4，归一化后 -4+2=-2，返回 -1，会得到 INVALID_PARAM.
    st = tensor_sum(x, out, -4, 0);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_reduce_errors_output_shape()
{
    TEST("tensor_reduce errors: output shape mismatch");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(data, 2, dims);

    // 输出形状与预期不符
    int wrong_dims[] = {2, 2};
    Tensor *out_wrong = tensor_create(2, wrong_dims);
    TensorStatus st = tensor_sum(x, out_wrong, 0, 0); // 预期输出 (3,) 但给了 (2,2)
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    // keepdims 情况下输出形状错误
    int wrong_keepdims[] = {1, 2};
    Tensor *out_keep_wrong = tensor_create(2, wrong_keepdims);
    st = tensor_sum(x, out_keep_wrong, 0, 1); // 预期 (1,3) 但给了 (1,2)
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    tensor_destroy(x);
    tensor_destroy(out_wrong);
    tensor_destroy(out_keep_wrong);
    PASS();
}

void test_quantile_errors()
{
    TEST("tensor_quantile errors");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *q = tensor_create(0, NULL); // 标量
    int out_dims[] = {2};
    Tensor *out = tensor_create(1, out_dims);

    // q 超出 [0,1]
    q->data[0] = -0.1f;
    TensorStatus st = tensor_quantile(x, q, 1, 0, QUANTILE_LINEAR, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    q->data[0] = 1.1f;
    st = tensor_quantile(x, q, 1, 0, QUANTILE_LINEAR, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    // q 不是标量（简单测试：传一个 1D 张量长度 >1，当前实现只支持标量）
    Tensor *q2 = tensor_create(1, (int[]){2}); // 长度为2
    q2->data[0] = 0.5f;
    q2->data[1] = 0.5f;
    st = tensor_quantile(x, q2, 1, 0, QUANTILE_LINEAR, out);
    assert(st == TENSOR_ERR_UNSUPPORTED); // 根据实现，返回 UNSUPPORTED

    tensor_destroy(x);
    tensor_destroy(q);
    tensor_destroy(q2);
    tensor_destroy(out);
    PASS();
}
void test_cummax()
{
    TEST("tensor_cummax");
    int dims[] = {2, 3};
    float data[] = {1, 5, 3, 4, 2, 6};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    // 沿轴0累积（跨行）
    tensor_cummax(x, 0, out);
    float expected_axis0[] = {1, 5, 3, 4, 5, 6}; // 第0列:1->1,1->4? 实际: (1,4) => [1,4]; 第1列:(5,2)=> [5,5]; 第2列:(3,6)=> [3,6]
    assert(check_tensor(out, expected_axis0, 6));

    // 沿轴1累积（跨列）
    tensor_cummax(x, 1, out);
    float expected_axis1[] = {1, 5, 5, 4, 4, 6}; // 第0行: [1,5,5]; 第1行: [4,4,6]
    assert(check_tensor(out, expected_axis1, 6));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_cummin()
{
    TEST("tensor_cummin");
    int dims[] = {2, 3};
    float data[] = {3, 1, 4, 2, 5, 0};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    // 沿轴0
    tensor_cummin(x, 0, out);
    float expected_axis0[] = {3, 1, 4, 2, 1, 0}; // 第0列:3->3, 3->2? 实际: (3,2)=> [3,2]; 第1列:(1,5)=> [1,1]; 第2列:(4,0)=> [4,0]
    assert(check_tensor(out, expected_axis0, 6));

    // 沿轴1
    tensor_cummin(x, 1, out);
    float expected_axis1[] = {3, 1, 1, 2, 2, 0}; // 第0行:[3,1,1]; 第1行:[2,2,0]
    assert(check_tensor(out, expected_axis1, 6));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_logcumsumexp()
{
    TEST("tensor_logcumsumexp");
    int dims[] = {2, 3};
    float data[] = {0, 1, 0, 2, 1, 0};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    // 沿轴0累积（跨行）
    tensor_logcumsumexp(x, 0, out);
    float expected_axis0[6];
    // 第0行（累积到行0）：就是原始值
    expected_axis0[0] = data[0];
    expected_axis0[1] = data[1];
    expected_axis0[2] = data[2];
    // 第1行（累积到行1）
    expected_axis0[3] = logf(expf(data[0]) + expf(data[3]));
    expected_axis0[4] = logf(expf(data[1]) + expf(data[4]));
    expected_axis0[5] = logf(expf(data[2]) + expf(data[5]));
    assert(check_tensor(out, expected_axis0, 6));

    // 沿轴1累积（跨列）
    tensor_logcumsumexp(x, 1, out);
    float expected_axis1[6];
    // 第0行
    expected_axis1[0] = data[0];
    expected_axis1[1] = logf(expf(data[0]) + expf(data[1]));
    expected_axis1[2] = logf(expf(data[0]) + expf(data[1]) + expf(data[2]));
    // 第1行
    expected_axis1[3] = data[3];
    expected_axis1[4] = logf(expf(data[3]) + expf(data[4]));
    expected_axis1[5] = logf(expf(data[3]) + expf(data[4]) + expf(data[5]));
    assert(check_tensor(out, expected_axis1, 6));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}
/* ==================== 主函数 ==================== */

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
    test_median();
    test_mode();
    test_quantile();

    // 错误测试
    test_reduce_errors_axis();
    test_reduce_errors_output_shape();
    test_quantile_errors();
    test_cummax();
    test_cummin();
    test_logcumsumexp();
    printf("All reduce_ops tests passed!\n");
    return 0;
}