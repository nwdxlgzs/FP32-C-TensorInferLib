#include "tensor.h"
#include "math_ops.h"
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

/* ==================== 一元运算测试 ==================== */

void test_neg()
{
    TEST("tensor_neg");
    int dims[] = {2, 3};
    float data[] = {1, -2, 3, -4, 5, -6};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    TensorStatus status = tensor_neg(x, out);
    assert(status == TENSOR_OK);
    float expected[] = {-1, 2, -3, 4, -5, 6};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_abs()
{
    TEST("tensor_abs");
    int dims[] = {2, 3};
    float data[] = {1, -2, 3, -4, 5, -6};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_abs(x, out);
    float expected[] = {1, 2, 3, 4, 5, 6};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_sqrt()
{
    TEST("tensor_sqrt");
    int dims[] = {2, 2};
    float data[] = {0, 4, 9, 16};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_sqrt(x, out);
    float expected[] = {0, 2, 3, 4};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_rsqrt()
{
    TEST("tensor_rsqrt");
    int dims[] = {3};
    float data[] = {1, 4, 9};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_rsqrt(x, out);
    float expected[] = {1, 0.5f, 1.0f / 3};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_exp_log()
{
    TEST("tensor_exp_log");
    int dims[] = {3};
    float data[] = {0, 1, 2};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *exp_out = tensor_create(1, dims);
    Tensor *log_out = tensor_create(1, dims);

    tensor_exp(x, exp_out);
    float exp_expected[] = {1, expf(1), expf(2)};
    assert(check_tensor(exp_out, exp_expected, 3));

    tensor_log(exp_out, log_out);
    assert(check_tensor(log_out, data, 3));

    tensor_destroy(x);
    tensor_destroy(exp_out);
    tensor_destroy(log_out);
    PASS();
}

void test_trig()
{
    TEST("tensor_sin_cos");
    int dims[] = {2};
    float data[] = {0, 3.1415926f / 2};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *sin_out = tensor_create(1, dims);
    Tensor *cos_out = tensor_create(1, dims);

    tensor_sin(x, sin_out);
    float sin_expected[] = {0, 1};
    assert(check_tensor(sin_out, sin_expected, 2));

    tensor_cos(x, cos_out);
    float cos_expected[] = {1, 0};
    assert(check_tensor(cos_out, cos_expected, 2));

    tensor_destroy(x);
    tensor_destroy(sin_out);
    tensor_destroy(cos_out);
    PASS();
}

void test_sign()
{
    TEST("tensor_sign");
    int dims[] = {5};
    float data[] = {-2, -0, 0, 3, 5};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_sign(x, out);
    float expected[] = {-1, 0, 0, 1, 1};
    assert(check_tensor(out, expected, 5));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_reciprocal()
{
    TEST("tensor_reciprocal");
    int dims[] = {3};
    float data[] = {1, 2, 4};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_reciprocal(x, out);
    float expected[] = {1, 0.5f, 0.25f};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

/* 一元运算批量测试入口 */
void test_unary_ops()
{
    test_neg();
    test_abs();
    test_sqrt();
    test_rsqrt();
    test_exp_log();
    test_trig();
    test_sign();
    test_reciprocal();
    /* 其他一元运算类似，可自行添加 */
}

/* ==================== 二元运算测试 ==================== */

void test_add()
{
    TEST("tensor_add");
    int dims_a[] = {2, 3};
    int dims_b[] = {2, 3};
    float data_a[] = {1, 2, 3, 4, 5, 6};
    float data_b[] = {6, 5, 4, 3, 2, 1};
    Tensor *a = tensor_from_array(data_a, 2, dims_a);
    Tensor *b = tensor_from_array(data_b, 2, dims_b);
    Tensor *out = tensor_create(2, dims_a);

    tensor_add(a, b, out);
    float expected[] = {7, 7, 7, 7, 7, 7};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_add_broadcast()
{
    TEST("tensor_add broadcast");
    int dims_a[] = {2, 1}; // 2x1
    int dims_b[] = {3};    // 3
    float data_a[] = {1, 2};
    float data_b[] = {10, 20, 30};
    Tensor *a = tensor_from_array(data_a, 2, dims_a);
    Tensor *b = tensor_from_array(data_b, 1, dims_b);
    Tensor *out = tensor_create(2, (int[]){2, 3}); // 预期形状 2x3

    tensor_add(a, b, out);
    float expected[] = {11, 21, 31, 12, 22, 32};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_sub()
{
    TEST("tensor_sub");
    int dims[] = {2, 2};
    float data_a[] = {5, 6, 7, 8};
    float data_b[] = {1, 2, 3, 4};
    Tensor *a = tensor_from_array(data_a, 2, dims);
    Tensor *b = tensor_from_array(data_b, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_sub(a, b, out);
    float expected[] = {4, 4, 4, 4};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_mul()
{
    TEST("tensor_mul");
    int dims[] = {3};
    float data_a[] = {1, 2, 3};
    float data_b[] = {2, 3, 4};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_mul(a, b, out);
    float expected[] = {2, 6, 12};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_div()
{
    TEST("tensor_div");
    int dims[] = {3};
    float data_a[] = {1, 2, 3};
    float data_b[] = {2, 2, 2};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_div(a, b, out);
    float expected[] = {0.5f, 1, 1.5f};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_maximum_minimum()
{
    TEST("tensor_maximum/minimum");
    int dims[] = {2, 2};
    float data_a[] = {1, 5, 3, 8};
    float data_b[] = {2, 4, 6, 7};
    Tensor *a = tensor_from_array(data_a, 2, dims);
    Tensor *b = tensor_from_array(data_b, 2, dims);
    Tensor *max_out = tensor_create(2, dims);
    Tensor *min_out = tensor_create(2, dims);

    tensor_maximum(a, b, max_out);
    float max_expected[] = {2, 5, 6, 8};
    assert(check_tensor(max_out, max_expected, 4));

    tensor_minimum(a, b, min_out);
    float min_expected[] = {1, 4, 3, 7};
    assert(check_tensor(min_out, min_expected, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(max_out);
    tensor_destroy(min_out);
    PASS();
}

void test_pow()
{
    TEST("tensor_pow");
    int dims[] = {3};
    float data_a[] = {2, 3, 4};
    float data_b[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_pow(a, b, out);
    float expected[] = {2, 9, 64};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_hypot()
{
    TEST("tensor_hypot");
    int dims[] = {3};
    float data_a[] = {3, 4, 5};
    float data_b[] = {4, 3, 12};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_hypot(a, b, out);
    float expected[] = {5, 5, 13};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_atan2()
{
    TEST("tensor_atan2");
    int dims[] = {2};
    float data_a[] = {1, 0};
    float data_b[] = {1, -1};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_atan2(a, b, out);
    float expected[] = {atan2f(1, 1), atan2f(0, -1)}; // 0.785, 3.1416
    assert(check_tensor(out, expected, 2));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ==================== 标量二元运算测试 ==================== */

void test_add_scalar()
{
    TEST("tensor_add_scalar");
    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};
    Tensor *a = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_add_scalar(a, 10, out);
    float expected[] = {11, 12, 13, 14};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

void test_mul_scalar()
{
    TEST("tensor_mul_scalar");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_mul_scalar(a, 2, out);
    float expected[] = {2, 4, 6};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* ==================== 三元运算测试 ==================== */

void test_clamp()
{
    TEST("tensor_clamp");
    int dims[] = {4};
    float data[] = {-5, 0, 3, 10};
    Tensor *x = tensor_from_array(data, 1, dims);
    float min_val = 0, max_val = 5;
    Tensor *min_t = tensor_wrap(&min_val, 0, NULL, NULL);
    Tensor *max_t = tensor_wrap(&max_val, 0, NULL, NULL);
    Tensor *out = tensor_create(1, dims);

    tensor_clamp(x, min_t, max_t, out);
    float expected[] = {0, 0, 3, 5};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(x);
    tensor_destroy(min_t);
    tensor_destroy(max_t);
    tensor_destroy(out);
    PASS();
}

void test_clamp_scalar()
{
    TEST("tensor_clamp_scalar");
    int dims[] = {4};
    float data[] = {-5, 0, 3, 10};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_clamp_scalar(x, 0, 5, out);
    float expected[] = {0, 0, 3, 5};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_where()
{
    TEST("tensor_where");
    int dims[] = {3};
    float cond_data[] = {1, 0, 1};
    float x_data[] = {10, 20, 30};
    float y_data[] = {100, 200, 300};
    Tensor *cond = tensor_from_array(cond_data, 1, dims);
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_from_array(y_data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_where(cond, x, y, out);
    float expected[] = {10, 200, 30};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(cond);
    tensor_destroy(x);
    tensor_destroy(y);
    tensor_destroy(out);
    PASS();
}

/* ==================== 主函数 ==================== */

int main()
{
    test_unary_ops();

    test_add();
    test_add_broadcast();
    test_sub();
    test_mul();
    test_div();
    test_maximum_minimum();
    test_pow();
    test_hypot();
    test_atan2();

    test_add_scalar();
    test_mul_scalar();

    test_clamp();
    test_clamp_scalar();
    test_where();

    printf("All math_ops tests passed!\n");
    return 0;
}