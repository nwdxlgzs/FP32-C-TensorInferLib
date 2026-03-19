#include "tensor.h"
#include "math_ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

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

/* ==================== 原有测试（保持不变） ==================== */

/* 一元运算测试 */
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

/* 二元运算测试 */
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

/* 标量二元运算测试 */
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

/* 三元运算测试 */
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

/* 特殊函数测试 */
void test_sigmoid()
{
    TEST("tensor_sigmoid");
    int dims[] = {4};
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_sigmoid(x, out);
    // 预期值由计算得出
    float expected[] = {
        1.0f / (1.0f + expf(2.0f)), // ~0.1192
        1.0f / (1.0f + expf(1.0f)), // ~0.2689
        0.5f,
        1.0f / (1.0f + expf(-1.0f)) // ~0.7311
    };
    assert(check_tensor(out, expected, 4));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_logit()
{
    TEST("tensor_logit");
    int dims[] = {3};
    float data[] = {0.1f, 0.5f, 0.9f};
    Tensor *p = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);
    float eps = 1e-6f;

    TensorStatus st = tensor_logit(p, eps, out);
    assert(st == TENSOR_OK);
    float expected[] = {
        logf(0.1f / 0.9f), // ~ -2.1972
        logf(0.5f / 0.5f), // 0.0
        logf(0.9f / 0.1f)  // ~ 2.1972
    };
    assert(check_tensor(out, expected, 3));

    // 测试剪裁：p=0.0 会被 eps 替换
    float data2[] = {0.0f, 0.5f, 1.0f};
    Tensor *p2 = tensor_from_array(data2, 1, dims);
    Tensor *out2 = tensor_create(1, dims);
    st = tensor_logit(p2, 1e-4f, out2);
    assert(st == TENSOR_OK);

    // 使用相对误差比较，因为数值较大时绝对误差可能超过 EPS
    float expected2[] = {
        logf(1e-4f / (1.0f - 1e-4f)), // 接近 -9.21
        0.0f,
        logf((1.0f - 1e-4f) / 1e-4f) // 接近 9.21
    };
    for (int i = 0; i < 3; i++)
    {
        float rel_diff = fabsf(out2->data[i] - expected2[i]) / fmaxf(1.0f, fabsf(expected2[i]));
        assert(rel_diff < 1e-4);
    }

    tensor_destroy(p);
    tensor_destroy(out);
    tensor_destroy(p2);
    tensor_destroy(out2);
    PASS();
}

void test_gamma()
{
    TEST("tensor_gamma");
    int dims[] = {4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_gamma(x, out);
    float expected[] = {
        tgammaf(1.0f), // 1
        tgammaf(2.0f), // 1
        tgammaf(3.0f), // 2
        tgammaf(4.0f)  // 6
    };
    assert(check_tensor(out, expected, 4));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_lgamma()
{
    TEST("tensor_lgamma");
    int dims[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_lgamma(x, out);
    float expected[] = {
        lgammaf(1.0f), // 0
        lgammaf(2.0f), // 0
        lgammaf(3.0f)  // ~0.6931
    };
    assert(check_tensor(out, expected, 3));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_bessel_i0()
{
    TEST("tensor_bessel_i0");
    int dims[] = {5};
    float data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_bessel_i0(x, out);
    // 使用已知近似值
    float expected[] = {
        1.0f,        // I0(0) = 1
        1.26606588f, // I0(1)
        2.27958530f, // I0(2)
        4.88079259f, // I0(3)
        11.30192195f // I0(4)
    };
    assert(check_tensor(out, expected, 5));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

/* ==================== 新增测试 ==================== */

/* 测试缺失的一元运算 */
void test_unary_missing()
{
    TEST("tensor_unary_missing (acos, asin, atan, ...)");
    int dims[] = {3};
    float data[] = {0.5f, 0.0f, -0.5f}; // 对于反三角，需要[-1,1]
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    // acos
    tensor_acos(x, out);
    float exp_acos[] = {acosf(0.5f), acosf(0.0f), acosf(-0.5f)};
    assert(check_tensor(out, exp_acos, 3));

    // asin
    tensor_asin(x, out);
    float exp_asin[] = {asinf(0.5f), asinf(0.0f), asinf(-0.5f)};
    assert(check_tensor(out, exp_asin, 3));

    // atan
    float data2[] = {1.0f, 0.0f, -1.0f};
    memcpy(x->data, data2, 3 * sizeof(float));
    tensor_atan(x, out);
    float exp_atan[] = {atanf(1.0f), atanf(0.0f), atanf(-1.0f)};
    assert(check_tensor(out, exp_atan, 3));

    // acosh (x>=1)
    float data3[] = {1.0f, 2.0f, 3.0f};
    memcpy(x->data, data3, 3 * sizeof(float));
    tensor_acosh(x, out);
    float exp_acosh[] = {acoshf(1.0f), acoshf(2.0f), acoshf(3.0f)};
    assert(check_tensor(out, exp_acosh, 3));

    // asinh
    float data4[] = {0.5f, 1.0f, -1.0f};
    memcpy(x->data, data4, 3 * sizeof(float));
    tensor_asinh(x, out);
    float exp_asinh[] = {asinhf(0.5f), asinhf(1.0f), asinhf(-1.0f)};
    assert(check_tensor(out, exp_asinh, 3));

    // atanh (|x|<1)
    float data5[] = {0.5f, 0.0f, -0.5f};
    memcpy(x->data, data5, 3 * sizeof(float));
    tensor_atanh(x, out);
    float exp_atanh[] = {atanhf(0.5f), atanhf(0.0f), atanhf(-0.5f)};
    assert(check_tensor(out, exp_atanh, 3));

    // ceil, floor, round, trunc
    float data6[] = {1.2f, -1.8f, 2.5f};
    memcpy(x->data, data6, 3 * sizeof(float));
    tensor_ceil(x, out);
    float exp_ceil[] = {2.0f, -1.0f, 3.0f};
    assert(check_tensor(out, exp_ceil, 3));
    tensor_floor(x, out);
    float exp_floor[] = {1.0f, -2.0f, 2.0f};
    assert(check_tensor(out, exp_floor, 3));
    tensor_round(x, out);
    float exp_round[] = {1.0f, -2.0f, 3.0f}; // 2.5 四舍五入到3
    assert(check_tensor(out, exp_round, 3));
    tensor_trunc(x, out);
    float exp_trunc[] = {1.0f, -1.0f, 2.0f};
    assert(check_tensor(out, exp_trunc, 3));

    // square, cube
    float data7[] = {2.0f, -3.0f, 4.0f};
    memcpy(x->data, data7, 3 * sizeof(float));
    tensor_square(x, out);
    float exp_square[] = {4.0f, 9.0f, 16.0f};
    assert(check_tensor(out, exp_square, 3));
    tensor_cube(x, out);
    float exp_cube[] = {8.0f, -27.0f, 64.0f};
    assert(check_tensor(out, exp_cube, 3));

    // erf, erfc
    float data8[] = {0.0f, 1.0f, -1.0f};
    memcpy(x->data, data8, 3 * sizeof(float));
    tensor_erf(x, out);
    float exp_erf[] = {erff(0.0f), erff(1.0f), erff(-1.0f)};
    assert(check_tensor(out, exp_erf, 3));
    tensor_erfc(x, out);
    float exp_erfc[] = {erfcf(0.0f), erfcf(1.0f), erfcf(-1.0f)};
    assert(check_tensor(out, exp_erfc, 3));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

/* 测试缺失的二元运算 */
void test_binary_missing()
{
    TEST("tensor_binary_missing (copysign, nextafter, fmod)");
    int dims[] = {3};
    float data_a[] = {1.0f, 2.0f, -3.0f};
    float data_b[] = {2.0f, -1.0f, 2.0f};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_copysign(a, b, out);
    float exp_copysign[] = {1.0f, -2.0f, 3.0f}; // 绝对值来自 a，符号来自 b
    assert(check_tensor(out, exp_copysign, 3));

    tensor_nextafter(a, b, out);
    // nextafter 结果取决于浮点，我们只检查非 NaN
    for (size_t i = 0; i < 3; ++i)
        assert(!isnan(out->data[i]));

    float data_c[] = {5.0f, 5.0f, 5.0f};
    float data_d[] = {3.0f, 3.0f, 3.0f};
    memcpy(a->data, data_c, 3 * sizeof(float));
    memcpy(b->data, data_d, 3 * sizeof(float));
    tensor_fmod(a, b, out);
    float exp_fmod[] = {2.0f, 2.0f, 2.0f};
    assert(check_tensor(out, exp_fmod, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* 测试缺失的标量二元运算 */
void test_scalar_missing()
{
    TEST("tensor_scalar_missing (sub_scalar, div_scalar, pow_scalar, max_scalar, min_scalar)");
    int dims[] = {3};
    float data[] = {2.0f, 4.0f, 8.0f};
    Tensor *a = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_sub_scalar(a, 1.0f, out);
    float exp_sub[] = {1.0f, 3.0f, 7.0f};
    assert(check_tensor(out, exp_sub, 3));

    tensor_div_scalar(a, 2.0f, out);
    float exp_div[] = {1.0f, 2.0f, 4.0f};
    assert(check_tensor(out, exp_div, 3));

    tensor_pow_scalar(a, 2.0f, out);
    float exp_pow[] = {4.0f, 16.0f, 64.0f};
    assert(check_tensor(out, exp_pow, 3));

    tensor_maximum_scalar(a, 5.0f, out);
    float exp_max[] = {5.0f, 5.0f, 8.0f};
    assert(check_tensor(out, exp_max, 3));

    tensor_minimum_scalar(a, 5.0f, out);
    float exp_min[] = {2.0f, 4.0f, 5.0f};
    assert(check_tensor(out, exp_min, 3));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* 测试形状不匹配错误 */
void test_shape_mismatch_errors()
{
    TEST("tensor_shape_mismatch_errors");
    int dims_a[] = {2, 3};
    int dims_b[] = {2, 2};
    float data[6] = {0};
    Tensor *a = tensor_from_array(data, 2, dims_a);
    Tensor *b = tensor_from_array(data, 2, dims_b);
    Tensor *out = tensor_create(2, dims_a); // 输出形状符合 a，但 a 和 b 不广播

    TensorStatus st = tensor_add(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    // 输出形状与广播结果不一致
    Tensor *out2 = tensor_create(2, dims_b);
    st = tensor_add(a, a, out2); // a+a 广播为 dims_a, 但 out2 是 dims_b
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    tensor_destroy(out2);
    PASS();
}

/* 测试除零情况（产生 Inf）*/
void test_div_by_zero()
{
    TEST("tensor_div_by_zero");
    int dims[] = {2};
    float data_a[] = {1.0f, 1.0f};
    float data_b[] = {0.0f, 0.0f};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_div(a, b, out);
    assert(isinf(out->data[0]) && isinf(out->data[1]));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* 测试 tan 等其他三角函数 */
void test_trig_more()
{
    TEST("tensor_tan, asin, acos");
    int dims[] = {3};
    float data[] = {0.0f, 0.5f, 1.0f};
    Tensor *x = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_tan(x, out);
    float exp_tan[] = {tanf(0.0f), tanf(0.5f), tanf(1.0f)};
    assert(check_tensor(out, exp_tan, 3));

    tensor_asin(x, out);
    float exp_asin[] = {asinf(0.0f), asinf(0.5f), asinf(1.0f)};
    assert(check_tensor(out, exp_asin, 3));

    tensor_acos(x, out);
    float exp_acos[] = {acosf(0.0f), acosf(0.5f), acosf(1.0f)};
    assert(check_tensor(out, exp_acos, 3));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}
void test_isclose()
{
    TEST("tensor_isclose");
    int dims[] = {4};
    float data_a[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float data_b[] = {1.0f, 1.0f + 1e-4f, 1.0f + 1e-3f, NAN};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    // 使用 rtol=1e-5, atol=1e-4
    tensor_isclose(a, b, 1e-5f, 1e-4f, out);
    float expected[] = {1.0f, 1.0f, 0.0f, 0.0f}; // NAN 与任何数比较应为 false
    assert(check_tensor(out, expected, 4));

    // 测试广播
    float data_scalar[] = {1.0f};
    Tensor *scalar = tensor_from_array(data_scalar, 0, NULL); // 0维标量
    Tensor *out2 = tensor_create(1, dims);
    tensor_isclose(a, scalar, 1e-5f, 1e-4f, out2);
    float expected2[] = {1.0f, 1.0f, 1.0f, 1.0f};
    assert(check_tensor(out2, expected2, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    tensor_destroy(scalar);
    tensor_destroy(out2);
    PASS();
}

void test_remainder()
{
    TEST("tensor_remainder");
    int dims[] = {5};
    float data_a[] = {5.0f, 5.0f, 5.0f, -5.0f, -5.0f};
    float data_b[] = {3.0f, -3.0f, 0.0f, 3.0f, -3.0f};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    TensorStatus st = tensor_remainder(a, b, out);
    assert(st == TENSOR_OK);
    // 预期值（IEEE 余数）：
    // 5 rem 3   = 5 - round(5/3)*3 = 5 - 2*3 = -1
    // 5 rem -3  = 5 - round(5/-3)*(-3) = 5 - (-2)*(-3) = 5 - 6 = -1
    // 5 rem 0   = 产生 NaN（或 Inf？根据标准应为 NaN）
    // -5 rem 3  = -5 - round(-5/3)*3 = -5 - (-2)*3 = -5 + 6 = 1
    // -5 rem -3 = -5 - round(-5/-3)*(-3) = -5 - 2*(-3) = -5 + 6 = 1
    float expected[] = {-1.0f, -1.0f, NAN, 1.0f, 1.0f};
    for (int i = 0; i < 5; ++i)
    {
        if (i == 2)
        {
            assert(isnan(out->data[i]));
        }
        else
        {
            assert(fabsf(out->data[i] - expected[i]) < 1e-5f);
        }
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}
/* ==================== 主函数 ==================== */

int main()
{
    // 原有测试
    test_neg();
    test_abs();
    test_sqrt();
    test_rsqrt();
    test_exp_log();
    test_trig();
    test_sign();
    test_reciprocal();

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

    test_sigmoid();
    test_logit();
    test_gamma();
    test_lgamma();
    test_bessel_i0();

    // 新增测试
    test_unary_missing();
    test_binary_missing();
    test_scalar_missing();
    test_shape_mismatch_errors();
    test_div_by_zero();
    test_trig_more();
    test_isclose();
    test_remainder();

    printf("All math_ops tests passed!\n");
    return 0;
}