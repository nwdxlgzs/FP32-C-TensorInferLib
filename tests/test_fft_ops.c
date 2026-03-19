#include "tensor.h"
#include "fft_ops.h"
#include "utils.h"
#include <stdio.h>
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

static int check_tensor(const Tensor *t, const float *expected, size_t n)
{
    if (tensor_size(t) != n)
        return 0;
    for (size_t i = 0; i < n; ++i)
        if (!approx_equal(t->data[i], expected[i], EPS))
            return 0;
    return 1;
}

/* ==================== 原有测试 ==================== */

void test_rfft_irfft_simple()
{
    TEST("rfft+irfft simple (power of two)");
    int dims[] = {8};
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *x = tensor_from_array(data, 1, dims);
    int out_dims[] = {5, 2};
    Tensor *out_rfft = tensor_create(2, out_dims);
    TensorStatus status = tensor_fft_rfft(x, out_rfft);
    assert(status == TENSOR_OK);

    float expected_rfft[] = {
        36.0f, 0.0f,
        -4.0f, 9.656854f,
        -4.0f, 4.0f,
        -4.0f, 1.656854f,
        -4.0f, 0.0f};
    assert(check_tensor(out_rfft, expected_rfft, 10));

    Tensor *out_irfft = tensor_create(1, dims);
    status = tensor_fft_irfft(out_rfft, 0, out_irfft);
    assert(status == TENSOR_OK);
    assert(check_tensor(out_irfft, data, 8));

    tensor_destroy(x);
    tensor_destroy(out_rfft);
    tensor_destroy(out_irfft);
    PASS();
}

void test_rfft_irfft_sine()
{
    TEST("rfft+irfft sine (power of two)");
    int n = 8;
    int dims[] = {n};
    float data[8];
    for (int i = 0; i < n; i++)
        data[i] = sinf(2 * M_PI * 2 * i / n);
    Tensor *x = tensor_from_array(data, 1, dims);

    int out_dims[] = {n / 2 + 1, 2};
    Tensor *out_rfft = tensor_create(2, out_dims);
    tensor_fft_rfft(x, out_rfft);

    Tensor *out_irfft = tensor_create(1, dims);
    tensor_fft_irfft(out_rfft, 0, out_irfft);

    assert(check_tensor(out_irfft, data, n));

    tensor_destroy(x);
    tensor_destroy(out_rfft);
    tensor_destroy(out_irfft);
    PASS();
}

void test_rfft_batch()
{
    TEST("rfft batch (power of two)");
    int dims[] = {2, 4};
    float data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8};
    Tensor *x = tensor_from_array(data, 2, dims);
    int out_dims[] = {2, 3, 2};
    Tensor *out_rfft = tensor_create(3, out_dims);
    tensor_fft_rfft(x, out_rfft);

    float expected[] = {
        10.0f, 0.0f, -2.0f, 2.0f, -2.0f, 0.0f,
        26.0f, 0.0f, -2.0f, 2.0f, -2.0f, 0.0f};
    assert(check_tensor(out_rfft, expected, 12));

    tensor_destroy(x);
    tensor_destroy(out_rfft);
    PASS();
}

void test_irfft_with_n()
{
    TEST("irfft with explicit n (even)");
    int dims[] = {5, 2};
    float data[] = {
        36.0f, 0.0f,
        -4.0f, 9.656854f,
        -4.0f, 4.0f,
        -4.0f, 1.656854f,
        -4.0f, 0.0f};
    Tensor *src = tensor_from_array(data, 2, dims);
    int out_dims[] = {8};
    Tensor *out = tensor_create(1, out_dims);
    tensor_fft_irfft(src, 8, out);
    float expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    assert(check_tensor(out, expected, 8));

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_rfft_irfft_non_power_of_two()
{
    TEST("rfft+irfft non-power-of-two (length 3)");
    int dims3[] = {3};
    float data3[] = {1.0f, 2.0f, 3.0f};
    Tensor *x3 = tensor_from_array(data3, 1, dims3);
    int out_dims3[] = {3 / 2 + 1, 2}; // 2,2
    Tensor *r3 = tensor_create(2, out_dims3);
    tensor_fft_rfft(x3, r3);
    Tensor *y3 = tensor_create(1, dims3);
    tensor_fft_irfft(r3, 3, y3);
    assert(check_tensor(y3, data3, 3));
    tensor_destroy(x3);
    tensor_destroy(r3);
    tensor_destroy(y3);

    TEST("rfft+irfft non-power-of-two (length 5)");
    int dims5[] = {5};
    float data5[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
    Tensor *x5 = tensor_from_array(data5, 1, dims5);
    int out_dims5[] = {5 / 2 + 1, 2}; // 3,2
    Tensor *r5 = tensor_create(2, out_dims5);
    tensor_fft_rfft(x5, r5);
    Tensor *y5 = tensor_create(1, dims5);
    tensor_fft_irfft(r5, 5, y5);
    assert(check_tensor(y5, data5, 5));
    tensor_destroy(x5);
    tensor_destroy(r5);
    tensor_destroy(y5);

    TEST("rfft+irfft non-power-of-two (length 6)");
    int dims6[] = {6};
    float data6[] = {1, 2, 3, 4, 5, 6};
    Tensor *x6 = tensor_from_array(data6, 1, dims6);
    int out_dims6[] = {6 / 2 + 1, 2}; // 4,2
    Tensor *r6 = tensor_create(2, out_dims6);
    tensor_fft_rfft(x6, r6);
    Tensor *y6 = tensor_create(1, dims6);
    tensor_fft_irfft(r6, 6, y6);
    assert(check_tensor(y6, data6, 6));
    tensor_destroy(x6);
    tensor_destroy(r6);
    tensor_destroy(y6);

    TEST("rfft+irfft non-power-of-two (length 7)");
    int dims7[] = {7};
    float data7[] = {0, 1, 2, 3, 4, 5, 6};
    Tensor *x7 = tensor_from_array(data7, 1, dims7);
    int out_dims7[] = {7 / 2 + 1, 2}; // 4,2
    Tensor *r7 = tensor_create(2, out_dims7);
    tensor_fft_rfft(x7, r7);
    Tensor *y7 = tensor_create(1, dims7);
    tensor_fft_irfft(r7, 7, y7);
    assert(check_tensor(y7, data7, 7));
    tensor_destroy(x7);
    tensor_destroy(r7);
    tensor_destroy(y7);
    PASS(); // 整体通过
}

/* ==================== 新增错误测试 ==================== */

void test_fft_errors_ndim()
{
    TEST("rfft error: ndim < 1");
    Tensor *x = tensor_create(0, NULL); // 标量，ndim=0
    int out_dims[] = {1, 2};
    Tensor *out = tensor_create(2, out_dims);
    TensorStatus st = tensor_fft_rfft(x, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(x);
    tensor_destroy(out);
    PASS();

    TEST("irfft error: ndim < 2");
    int dims[] = {5}; // 1D 张量，需要至少2维
    float data[] = {1, 2, 3, 4, 5};
    Tensor *src = tensor_from_array(data, 1, dims);
    int out_dims2[] = {8};
    Tensor *out2 = tensor_create(1, out_dims2);
    st = tensor_fft_irfft(src, 8, out2);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(src);
    tensor_destroy(out2);
    PASS();
}

void test_fft_errors_shape_mismatch()
{
    TEST("rfft error: output shape mismatch");
    int dims_in[] = {8};
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *x = tensor_from_array(data, 1, dims_in);

    int wrong_dims[] = {4, 2}; // 应该为 {5,2}
    Tensor *out = tensor_create(2, wrong_dims);
    TensorStatus st = tensor_fft_rfft(x, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(x);
    tensor_destroy(out);

    TEST("irfft error: output shape mismatch");
    int dims_src[] = {5, 2};
    float src_data[] = {36, 0, -4, 9.656854, -4, 4, -4, 1.656854, -4, 0};
    Tensor *src = tensor_from_array(src_data, 2, dims_src);
    int wrong_out_dims[] = {9}; // 应 {8}
    Tensor *out2 = tensor_create(1, wrong_out_dims);
    st = tensor_fft_irfft(src, 8, out2);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(src);
    tensor_destroy(out2);
    PASS();
}

void test_irfft_error_n_mismatch()
{
    TEST("irfft error: n mismatch with input shape");
    int dims_src[] = {5, 2}; // 对应偶数 n=8 或奇数 n=9，传入 n=7 应不匹配
    float src_data[10];
    Tensor *src = tensor_from_array(src_data, 2, dims_src);
    int out_dims[] = {7};
    Tensor *out = tensor_create(1, out_dims);
    TensorStatus st = tensor_fft_irfft(src, 7, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_irfft_error_last_dim_not_2()
{
    TEST("irfft error: last dimension != 2");
    int dims_src[] = {5, 3}; // 最后一维应为2
    float src_data[15];
    Tensor *src = tensor_from_array(src_data, 2, dims_src);
    int out_dims[] = {8};
    Tensor *out = tensor_create(1, out_dims);
    TensorStatus st = tensor_fft_irfft(src, 8, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_fft_noncontiguous_input()
{
    TEST("rfft with non-contiguous input");
    int dims[] = {2, 4};
    float data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8};
    Tensor *a = tensor_from_array(data, 2, dims);
    int trans_dims[] = {4, 2};
    int strides[] = {1, 4};
    Tensor *noncontig = tensor_view(a, 2, trans_dims, strides);
    assert(noncontig != NULL);

    // 对非连续版本进行 rfft
    int out_dims_nc[] = {4, 2, 2};
    Tensor *out_nc = tensor_create(3, out_dims_nc);
    tensor_fft_rfft(noncontig, out_nc);

    // 创建 noncontig 的逻辑连续副本
    Tensor *cont_copy = tensor_create(2, trans_dims);
    tensor_copy(cont_copy, noncontig); // 正确复制逻辑数据
    Tensor *out_cont_nc = tensor_create(3, out_dims_nc);
    tensor_fft_rfft(cont_copy, out_cont_nc);

    // 比较两者
    assert(tensor_allclose(out_nc, out_cont_nc, 1e-5, 1e-5));

    tensor_destroy(a);
    tensor_destroy(noncontig);
    tensor_destroy(cont_copy);
    tensor_destroy(out_nc);
    tensor_destroy(out_cont_nc);
    PASS();
}

/* ==================== 边界情况测试 ==================== */

void test_fft_length_one()
{
    TEST("rfft+irfft length 1");
    int dims[] = {1};
    float data[] = {3.14f};
    Tensor *x = tensor_from_array(data, 1, dims);
    int out_dims[] = {1, 2};
    Tensor *r = tensor_create(2, out_dims);
    tensor_fft_rfft(x, r);
    // 预期：实部为和，虚部0
    float exp_r[] = {3.14f, 0.0f};
    assert(check_tensor(r, exp_r, 2));
    Tensor *y = tensor_create(1, dims);
    tensor_fft_irfft(r, 1, y);
    assert(check_tensor(y, data, 1));
    tensor_destroy(x);
    tensor_destroy(r);
    tensor_destroy(y);
    PASS();
}

/* ==================== 新增复数到复数FFT/IFFT测试 ==================== */

void test_fft_ifft_simple()
{
    TEST("fft+ifft simple (power of two)");
    int n = 4;
    int dims[] = {n, 2}; // 复数张量
    // 构造信号： [1+0j, 0+1j, 0+0j, 0+0j]
    float data[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        0.0f, 0.0f};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *y = tensor_create(2, dims); // 用于存放FFT结果

    tensor_fft(x, y);

    // 手动计算DFT:
    float expected[] = {
        1.0f, 1.0f,
        2.0f, 0.0f,
        1.0f, -1.0f,
        0.0f, 0.0f};
    assert(check_tensor(y, expected, 8));

    // 逆变换还原
    Tensor *z = tensor_create(2, dims);
    tensor_ifft(y, z);
    // 预期还原回原始数据（IFFT已除以n）
    assert(check_tensor(z, data, 8));

    tensor_destroy(x);
    tensor_destroy(y);
    tensor_destroy(z);
    PASS();
}

void test_fft_ifft_sine()
{
    TEST("fft+ifft sine wave (power of two)");
    int n = 8;
    int k0 = 2;
    int dims[] = {n, 2};
    float data[16];
    for (int i = 0; i < n; i++)
    {
        float angle = 2 * M_PI * k0 * i / n;
        data[2 * i] = cosf(angle);
        data[2 * i + 1] = sinf(angle);
    }
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *y = tensor_create(2, dims);
    tensor_fft(x, y);

    // 预期：在k0处值为n，其他地方为0
    float expected[16];
    memset(expected, 0, sizeof(expected));
    expected[2 * k0] = n;     // 实部
    expected[2 * k0 + 1] = 0; // 虚部

    assert(tensor_allclose(y, tensor_from_array(expected, 2, dims), 1e-5, 1e-5));

    // 逆变换
    Tensor *z = tensor_create(2, dims);
    tensor_ifft(y, z);
    assert(tensor_allclose(z, x, 1e-5, 1e-5));

    tensor_destroy(x);
    tensor_destroy(y);
    tensor_destroy(z);
    PASS();
}

void test_fft_batch()
{
    TEST("fft batch (power of two)");
    int batch = 2;
    int n = 4;
    int dims[] = {batch, n, 2};
    // 第一个batch：信号1
    // 第二个batch：信号2
    float data[] = {
        // batch0: [1+0j, 0+1j, 0+0j, 0+0j]
        1, 0, 0, 1, 0, 0, 0, 0,
        // batch1: [1,1,1,1] 实部全1，虚部0
        1, 0, 1, 0, 1, 0, 1, 0};
    Tensor *x = tensor_from_array(data, 3, dims);
    Tensor *y = tensor_create(3, dims);
    tensor_fft(x, y);

    // 预期batch0结果同前
    float exp_batch0[] = {1, 1, 2, 0, 1, -1, 0, 0};
    // batch1: 直流信号，X[0]=4, 其余0
    float exp_batch1[] = {4, 0, 0, 0, 0, 0, 0, 0};
    float expected[2 * 8];
    memcpy(expected, exp_batch0, 8 * sizeof(float));
    memcpy(expected + 8, exp_batch1, 8 * sizeof(float));

    assert(tensor_allclose(y, tensor_from_array(expected, 3, dims), 1e-5, 1e-5));

    // 逆变换
    Tensor *z = tensor_create(3, dims);
    tensor_ifft(y, z);
    assert(tensor_allclose(z, x, 1e-5, 1e-5));

    tensor_destroy(x);
    tensor_destroy(y);
    tensor_destroy(z);
    PASS();
}

void test_fft_non_power_of_two()
{
    TEST("fft+ifft non-power-of-two (length 3)");
    int n = 3;
    int dims[] = {n, 2};
    // 构造简单信号： [1+0j, 1+0j, 1+0j] 直流
    float data[] = {1, 0, 1, 0, 1, 0};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *y = tensor_create(2, dims);
    tensor_fft(x, y);
    // 预期：X[0]=3, 其他0
    float exp_dc[] = {3, 0, 0, 0, 0, 0};
    assert(check_tensor(y, exp_dc, 6));

    Tensor *z = tensor_create(2, dims);
    tensor_ifft(y, z);
    assert(check_tensor(z, data, 6));
    tensor_destroy(x);
    tensor_destroy(y);
    tensor_destroy(z);

    // 长度5
    TEST("fft+ifft non-power-of-two (length 5)");
    n = 5;
    int dims5[] = {n, 2};
    float data5[10];
    for (int i = 0; i < 5; i++)
    {
        data5[2 * i] = i + 1; // 实部 1..5
        data5[2 * i + 1] = 0; // 虚部0
    }
    Tensor *x5 = tensor_from_array(data5, 2, dims5);
    Tensor *y5 = tensor_create(2, dims5);
    tensor_fft(x5, y5);
    Tensor *z5 = tensor_create(2, dims5);
    tensor_ifft(y5, z5);
    assert(tensor_allclose(z5, x5, 1e-5, 1e-5));
    tensor_destroy(x5);
    tensor_destroy(y5);
    tensor_destroy(z5);
    PASS();
}

void test_fft_errors_c2c()
{
    TEST("fft error: ndim < 2");
    int dims1[] = {4};
    float data[4] = {1, 2, 3, 4};
    Tensor *x1 = tensor_from_array(data, 1, dims1);
    int dims_out[] = {4, 2};
    Tensor *out1 = tensor_create(2, dims_out);
    TensorStatus st = tensor_fft(x1, out1);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(x1);
    tensor_destroy(out1);
    PASS();

    TEST("fft error: last dimension != 2");
    int dims2[] = {4, 3};
    Tensor *x2 = tensor_create(2, dims2);
    int dims_out2[] = {4, 3};
    Tensor *out2 = tensor_create(2, dims_out2);
    st = tensor_fft(x2, out2);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(x2);
    tensor_destroy(out2);
    PASS();

    TEST("fft error: output shape mismatch");
    int dims3[] = {4, 2};
    Tensor *x3 = tensor_create(2, dims3);
    int wrong_dims[] = {5, 2};
    Tensor *out3 = tensor_create(2, wrong_dims);
    st = tensor_fft(x3, out3);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(x3);
    tensor_destroy(out3);
    PASS();
}

void test_ifft_errors_c2c()
{
    TEST("ifft error: ndim < 2");
    int dims1[] = {4};
    float data[4] = {1, 2, 3, 4};
    Tensor *x1 = tensor_from_array(data, 1, dims1);
    int dims_out[] = {4, 2};
    Tensor *out1 = tensor_create(2, dims_out);
    TensorStatus st = tensor_ifft(x1, out1);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(x1);
    tensor_destroy(out1);
    PASS();

    TEST("ifft error: last dimension != 2");
    int dims2[] = {4, 3};
    Tensor *x2 = tensor_create(2, dims2);
    int dims_out2[] = {4, 3};
    Tensor *out2 = tensor_create(2, dims_out2);
    st = tensor_ifft(x2, out2);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(x2);
    tensor_destroy(out2);
    PASS();
}

void test_fft_noncontiguous_c2c()
{
    TEST("fft with non-contiguous input (complex)");
    int dims[] = {2, 4, 2}; // [batch, n, 2]
    float data[16];
    for (int i = 0; i < 16; i++)
        data[i] = (float)i;
    Tensor *a = tensor_from_array(data, 3, dims);
    // 创建一个视图，交换前两维
    int perm_dims[] = {4, 2, 2}; // [n, batch, 2]
    // 原连续 strides: [8,2,1] (因为 dims[0]=2, dims[1]=4, dims[2]=2)
    int new_strides[] = {2, 8, 1};
    Tensor *noncontig = tensor_view(a, 3, perm_dims, new_strides);
    assert(noncontig != NULL);

    // 对非连续张量进行fft
    Tensor *out_nc = tensor_create(3, perm_dims);
    tensor_fft(noncontig, out_nc);

    // 创建逻辑连续的副本
    Tensor *cont_copy = tensor_create(3, perm_dims);
    tensor_copy(cont_copy, noncontig);
    Tensor *out_cont = tensor_create(3, perm_dims);
    tensor_fft(cont_copy, out_cont);

    assert(tensor_allclose(out_nc, out_cont, 1e-5, 1e-5));

    tensor_destroy(a);
    tensor_destroy(noncontig);
    tensor_destroy(cont_copy);
    tensor_destroy(out_nc);
    tensor_destroy(out_cont);
    PASS();
}

void test_fft_length_one_c2c()
{
    TEST("fft+ifft length 1 (complex)");
    int dims[] = {1, 2};
    float data[] = {3.14f, 0.0f}; // 实部3.14
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *y = tensor_create(2, dims);
    tensor_fft(x, y);
    // FFT of length 1 should be same value
    assert(check_tensor(y, data, 2));
    Tensor *z = tensor_create(2, dims);
    tensor_ifft(y, z);
    assert(check_tensor(z, data, 2));
    tensor_destroy(x);
    tensor_destroy(y);
    tensor_destroy(z);
    PASS();
}

/* ==================== 主函数 ==================== */

int main()
{
    test_rfft_irfft_simple();
    test_rfft_irfft_sine();
    test_rfft_batch();
    test_irfft_with_n();
    test_rfft_irfft_non_power_of_two();

    test_fft_errors_ndim();
    test_fft_errors_shape_mismatch();
    test_irfft_error_n_mismatch();
    test_irfft_error_last_dim_not_2();
    test_fft_noncontiguous_input();

    test_fft_length_one();

    // 新增复数FFT/IFFT测试
    test_fft_ifft_simple();
    test_fft_ifft_sine();
    test_fft_batch();
    test_fft_non_power_of_two();
    test_fft_errors_c2c();
    test_ifft_errors_c2c();
    test_fft_noncontiguous_c2c();
    test_fft_length_one_c2c();

    printf("All fft_ops tests passed!\n");
    return 0;
}