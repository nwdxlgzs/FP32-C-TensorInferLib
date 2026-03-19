#include "tensor.h"
#include "nn_ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

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
    for (size_t i = 0; i < n; i++)
        if (!approx_equal(t->data[i], expected[i], EPS))
            return 0;
    return 1;
}

static void print_tensor(const Tensor *t, const char *name)
{
    printf("%s: ", name);
    for (size_t i = 0; i < t->size; i++)
        printf("%.6f ", t->data[i]);
    printf("\n");
}

/* ---------- 原有测试（保持不变） ---------- */
void test_conv1d()
{
    TEST("tensor_conv1d");
    float in_data[] = {1, 2, 3, 4, 5};
    float w_data[] = {1, 0, -1};
    float bias_val = 0;
    int in_dims[] = {1, 1, 5};
    int w_dims[] = {1, 1, 3};
    int out_dims[] = {1, 1, 3};
    Tensor *input = tensor_from_array(in_data, 3, in_dims);
    Tensor *weight = tensor_from_array(w_data, 3, w_dims);
    Tensor *bias = tensor_wrap(&bias_val, 0, NULL, NULL);
    Tensor *output = tensor_create(3, out_dims);

    ConvParams params = {0};
    params.pad[0] = 0;
    params.stride[0] = 1;
    params.dilation[0] = 1;
    params.groups = 1;

    TensorStatus status = tensor_conv1d(input, weight, bias, params, output);
    assert(status == TENSOR_OK);

    float expected[] = {-2, -2, -2};
    assert(check_tensor(output, expected, 3));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(bias);
    tensor_destroy(output);
    PASS();
}

void test_conv2d()
{
    TEST("tensor_conv2d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float w_data[] = {1, 0, 0, 1};
    int in_dims[] = {1, 1, 3, 3};
    int w_dims[] = {1, 1, 2, 2};
    int out_dims[] = {1, 1, 2, 2};
    Tensor *input = tensor_from_array(in_data, 4, in_dims);
    Tensor *weight = tensor_from_array(w_data, 4, w_dims);
    Tensor *output = tensor_create(4, out_dims);

    ConvParams params = {0};
    params.pad[0] = 0;
    params.pad[1] = 0;
    params.stride[0] = 1;
    params.stride[1] = 1;
    params.dilation[0] = 1;
    params.dilation[1] = 1;
    params.groups = 1;

    tensor_conv2d(input, weight, NULL, params, output);
    float expected[] = {6, 8, 12, 14};
    assert(check_tensor(output, expected, 4));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_conv3d()
{
    TEST("tensor_conv3d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float w_data[] = {1, 0, 0, 1, 0, 1, 1, 0};
    int in_dims[] = {1, 1, 2, 2, 2};
    int w_dims[] = {1, 1, 2, 2, 2};
    int out_dims[] = {1, 1, 1, 1, 1};
    Tensor *input = tensor_from_array(in_data, 5, in_dims);
    Tensor *weight = tensor_from_array(w_data, 5, w_dims);
    Tensor *output = tensor_create(5, out_dims);

    ConvParams params = {0};
    for (int i = 0; i < 3; i++)
    {
        params.pad[i] = 0;
        params.stride[i] = 1;
        params.dilation[i] = 1;
    }
    params.groups = 1;

    tensor_conv3d(input, weight, NULL, params, output);
    float expected[] = {18};
    assert(check_tensor(output, expected, 1));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_pool1d()
{
    TEST("tensor_pool1d");
    float in_data[] = {1, 3, 2, 4, 5};
    int in_dims[] = {1, 1, 5};
    Tensor *input = tensor_from_array(in_data, 3, in_dims);

    PoolParams params0 = {0};
    params0.kernel[0] = 2;
    params0.stride[0] = 2;
    params0.pad[0] = 0;
    params0.ceil_mode = 0;
    params0.count_include_pad = 0;

    int out_dims0[] = {1, 1, 2};
    Tensor *output0 = tensor_create(3, out_dims0);
    tensor_pool1d(input, POOL_MAX, params0, output0);
    float expected_max0[] = {3, 4};
    assert(check_tensor(output0, expected_max0, 2));

    tensor_pool1d(input, POOL_AVG, params0, output0);
    float expected_avg0[] = {2, 3};
    assert(check_tensor(output0, expected_avg0, 2));

    PoolParams params1 = {0};
    params1.kernel[0] = 2;
    params1.stride[0] = 2;
    params1.pad[0] = 0;
    params1.ceil_mode = 1;
    params1.count_include_pad = 0;

    int out_dims1[] = {1, 1, 3};
    Tensor *output1 = tensor_create(3, out_dims1);
    tensor_pool1d(input, POOL_MAX, params1, output1);
    float expected_max1[] = {3, 4, 5};
    assert(check_tensor(output1, expected_max1, 3));

    tensor_pool1d(input, POOL_AVG, params1, output1);
    float expected_avg1[] = {2, 3, 5};
    assert(check_tensor(output1, expected_avg1, 3));

    tensor_destroy(input);
    tensor_destroy(output0);
    tensor_destroy(output1);
    PASS();
}

void test_pool2d()
{
    TEST("tensor_pool2d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int in_dims[] = {1, 1, 3, 3};
    Tensor *input = tensor_from_array(in_data, 4, in_dims);
    int out_dims[] = {1, 1, 2, 2};
    Tensor *output = tensor_create(4, out_dims);

    PoolParams params = {0};
    params.kernel[0] = 2;
    params.kernel[1] = 2;
    params.stride[0] = 1;
    params.stride[1] = 1;
    params.pad[0] = 0;
    params.pad[1] = 0;

    tensor_pool2d(input, POOL_MAX, params, output);
    float expected_max[] = {5, 6, 8, 9};
    assert(check_tensor(output, expected_max, 4));

    tensor_pool2d(input, POOL_AVG, params, output);
    float expected_avg[] = {3, 4, 6, 7};
    assert(check_tensor(output, expected_avg, 4));

    tensor_destroy(input);
    tensor_destroy(output);
    PASS();
}

void test_pool3d()
{
    TEST("tensor_pool3d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int in_dims[] = {1, 1, 2, 2, 2};
    Tensor *input = tensor_from_array(in_data, 5, in_dims);
    int out_dims[] = {1, 1, 1, 1, 1};
    Tensor *output = tensor_create(5, out_dims);

    PoolParams params = {0};
    params.kernel[0] = 2;
    params.kernel[1] = 2;
    params.kernel[2] = 2;
    params.stride[0] = 1;
    params.stride[1] = 1;
    params.stride[2] = 1;
    params.pad[0] = 0;
    params.pad[1] = 0;
    params.pad[2] = 0;

    tensor_pool3d(input, POOL_MAX, params, output);
    assert(approx_equal(output->data[0], 8, EPS));

    tensor_pool3d(input, POOL_AVG, params, output);
    assert(approx_equal(output->data[0], 4.5, EPS));

    tensor_destroy(input);
    tensor_destroy(output);
    PASS();
}

void test_global_pool2d()
{
    TEST("tensor_global_avg/max_pool2d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int in_dims[] = {2, 1, 2, 2};
    Tensor *input = tensor_from_array(in_data, 4, in_dims);
    int out_dims[] = {2, 1, 1, 1};
    Tensor *avg_out = tensor_create(4, out_dims);
    Tensor *max_out = tensor_create(4, out_dims);

    tensor_global_avg_pool2d(input, avg_out);
    tensor_global_max_pool2d(input, max_out);

    float expected_avg[] = {2.5, 6.5};
    float expected_max[] = {4, 8};
    assert(check_tensor(avg_out, expected_avg, 2));
    assert(check_tensor(max_out, expected_max, 2));

    tensor_destroy(input);
    tensor_destroy(avg_out);
    tensor_destroy(max_out);
    PASS();
}

void test_batchnorm()
{
    TEST("tensor_batchnorm");
    float x_data[] = {1, 2, 3, 4};
    int x_dims[] = {2, 2, 1, 1};
    float mean_data[] = {1.5f, 3.5f};
    float var_data[] = {0.25f, 0.25f};
    float scale_data[] = {2, 2};
    float bias_data[] = {1, 1};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *mean = tensor_from_array(mean_data, 1, (int[]){2});
    Tensor *var = tensor_from_array(var_data, 1, (int[]){2});
    Tensor *scale = tensor_from_array(scale_data, 1, (int[]){2});
    Tensor *bias = tensor_from_array(bias_data, 1, (int[]){2});
    Tensor *y = tensor_create(4, x_dims);

    tensor_batchnorm(x, mean, var, scale, bias, 1e-12f, y);
    float expected[] = {-1, -5, 7, 3};
    assert(check_tensor(y, expected, 4));

    tensor_destroy(x);
    tensor_destroy(mean);
    tensor_destroy(var);
    tensor_destroy(scale);
    tensor_destroy(bias);
    tensor_destroy(y);
    PASS();
}

void test_layernorm()
{
    TEST("tensor_layernorm");
    float x_data[] = {1, 2, 3, 4, 5, 6};
    int x_dims[] = {2, 3};
    Tensor *x = tensor_from_array(x_data, 2, x_dims);
    Tensor *y = tensor_create(2, x_dims);

    tensor_layernorm(x, NULL, NULL, 1e-5f, y);
    float expected[] = {-1.22474487f, 0, 1.22474487f,
                        -1.22474487f, 0, 1.22474487f};
    assert(check_tensor(y, expected, 6));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_instancenorm()
{
    TEST("tensor_instancenorm");
    float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int x_dims[] = {2, 2, 2, 1};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, x_dims);

    tensor_instancenorm(x, NULL, NULL, 1e-12f, y);
    float expected[] = {-1, 1, -1, 1, -1, 1, -1, 1};
    assert(check_tensor(y, expected, 8));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_groupnorm()
{
    TEST("tensor_groupnorm");
    float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int x_dims[] = {1, 4, 2, 1};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, x_dims);

    tensor_groupnorm(x, NULL, NULL, 2, 1e-12f, y);
    float expected[] = {
        -1.3416407865f, -0.4472135955f, 0.4472135955f, 1.3416407865f,
        -1.3416407865f, -0.4472135955f, 0.4472135955f, 1.3416407865f};
    assert(check_tensor(y, expected, 8));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_lrn()
{
    TEST("tensor_lrn");
    float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int x_dims[] = {1, 4, 2, 1};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, x_dims);

    tensor_lrn(x, 3, 0.001f, 0.75f, 1.0f, y);
    assert(y != NULL);
    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_activations()
{
    TEST("activation functions");
    float x_data[] = {-2, -1, 0, 1, 2};
    int dims[] = {5};
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_create(1, dims);

    tensor_relu(x, y);
    float relu_exp[] = {0, 0, 0, 1, 2};
    assert(check_tensor(y, relu_exp, 5));

    tensor_leaky_relu(x, 0.01f, y);
    float leaky_exp[] = {-0.02f, -0.01f, 0, 1, 2};
    assert(check_tensor(y, leaky_exp, 5));

    tensor_elu(x, 1.0f, y);
    float elu_exp[] = {-0.86466f, -0.63212f, 0, 1, 2};
    assert(check_tensor(y, elu_exp, 5));

    tensor_selu(x, 1.67326f, 1.0507f, y);
    float selu_exp[] = {-1.520166f, -1.111331f, 0, 1.050701f, 2.101402f};
    assert(check_tensor(y, selu_exp, 5));

    tensor_gelu(x, y);
    tensor_swish(x, y);
    tensor_mish(x, y);
    tensor_softplus(x, y);
    tensor_softsign(x, y);
    tensor_hardswish(x, y);
    tensor_hardsigmoid(x, y);
    assert(y != NULL);

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_prelu()
{
    TEST("tensor_prelu");
    float x_data[] = {-2, -1, 0, 1, 2, -3, -2, -1, 0, 1};
    int x_dims[] = {1, 2, 1, 5};
    float alpha_data[] = {0.1f, 0.5f};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *alpha = tensor_from_array(alpha_data, 1, (int[]){2});
    Tensor *y = tensor_create(4, x_dims);

    tensor_prelu(x, alpha, y);
    float expected[] = {-0.2f, -0.1f, 0, 1, 2,
                        -1.5f, -1.0f, -0.5f, 0, 1};
    assert(check_tensor(y, expected, 10));

    tensor_destroy(x);
    tensor_destroy(alpha);
    tensor_destroy(y);
    PASS();
}

void test_linear()
{
    TEST("tensor_linear");
    float in_data[] = {1, 2, 3, 4, 5, 6};
    float w_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    float b_data[] = {0, 0, 0, 1};
    int in_dims[] = {2, 3};
    int w_dims[] = {4, 3};
    int out_dims[] = {2, 4};
    Tensor *input = tensor_from_array(in_data, 2, in_dims);
    Tensor *weight = tensor_from_array(w_data, 2, w_dims);
    Tensor *bias = tensor_from_array(b_data, 1, (int[]){4});
    Tensor *output = tensor_create(2, out_dims);

    tensor_linear(input, weight, bias, output);
    float expected[] = {1, 2, 3, 7, 4, 5, 6, 16};
    assert(check_tensor(output, expected, 8));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(bias);
    tensor_destroy(output);
    PASS();
}

void test_dropout()
{
    TEST("tensor_dropout");
    float x_data[] = {1, 2, 3, 4};
    int dims[] = {4};
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_create(1, dims);

    tensor_dropout(x, 0.5, 0, y);
    assert(check_tensor(y, x_data, 4));

    srand(0);
    tensor_dropout(x, 0.5, 1, y);
    assert(y->size == 4);

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_softmax()
{
    TEST("tensor_softmax");
    float x_data[] = {1, 2, 3};
    int dims[] = {3};
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_create(1, dims);

    tensor_softmax(x, 0, y);
    float exp_sum = expf(1) + expf(2) + expf(3);
    float expected[] = {expf(1) / exp_sum, expf(2) / exp_sum, expf(3) / exp_sum};
    assert(check_tensor(y, expected, 3));

    tensor_log_softmax(x, 0, y);
    float log_sum = logf(exp_sum);
    float expected_log[] = {1 - log_sum, 2 - log_sum, 3 - log_sum};
    assert(check_tensor(y, expected_log, 3));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_upsample2d_nearest()
{
    TEST("tensor_upsample2d nearest");
    float x_data[] = {1, 2, 3, 4};
    int x_dims[] = {1, 1, 2, 2};
    int out_dims[] = {1, 1, 4, 4};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, out_dims);

    tensor_upsample2d(x, 2, 2, UPSAMPLE_NEAREST, 0, y);
    float expected[] = {
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4};
    assert(check_tensor(y, expected, 16));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_upsample2d_linear()
{
    TEST("tensor_upsample2d linear with align_corners");
    {
        int in_dims[] = {1, 1, 2, 2};
        float in_data[] = {1, 2, 3, 4};
        Tensor *input = tensor_from_array(in_data, 4, in_dims);
        int out_dims[] = {1, 1, 4, 4};
        Tensor *out = tensor_create(4, out_dims);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_LINEAR, 0, out);
        float expected[] = {
            1.00f, 1.25f, 1.75f, 2.00f,
            1.50f, 1.75f, 2.25f, 2.50f,
            2.50f, 2.75f, 3.25f, 3.50f,
            3.00f, 3.25f, 3.75f, 4.00f};
        assert(check_tensor(out, expected, 16));
        tensor_destroy(input);
        tensor_destroy(out);
    }
    {
        int in_dims[] = {1, 1, 2, 2};
        float in_data[] = {1, 2, 3, 4};
        Tensor *input = tensor_from_array(in_data, 4, in_dims);
        int out_dims[] = {1, 1, 4, 4};
        Tensor *out = tensor_create(4, out_dims);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_LINEAR, 1, out);
        float expected[] = {
            1.0f, 4.0f / 3.0f, 5.0f / 3.0f, 2.0f,
            5.0f / 3.0f, 2.0f, 7.0f / 3.0f, 8.0f / 3.0f,
            7.0f / 3.0f, 8.0f / 3.0f, 3.0f, 10.0f / 3.0f,
            3.0f, 10.0f / 3.0f, 11.0f / 3.0f, 4.0f};
        assert(check_tensor(out, expected, 16));
        tensor_destroy(input);
        tensor_destroy(out);
    }
    {
        int in_dims[] = {1, 1, 2, 2};
        float in_data[] = {5.0f, 5.0f, 5.0f, 5.0f};
        Tensor *input = tensor_from_array(in_data, 4, in_dims);
        int out_dims[] = {1, 1, 4, 4};
        Tensor *out0 = tensor_create(4, out_dims);
        Tensor *out1 = tensor_create(4, out_dims);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_LINEAR, 0, out0);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_LINEAR, 1, out1);
        for (size_t i = 0; i < out0->size; i++)
        {
            assert(approx_equal(out0->data[i], 5.0f, EPS));
            assert(approx_equal(out1->data[i], 5.0f, EPS));
        }
        tensor_destroy(input);
        tensor_destroy(out0);
        tensor_destroy(out1);
    }
    PASS();
}

void test_upsample2d_cubic()
{
    TEST("tensor_upsample2d cubic");
    {
        int in_dims1[] = {1, 1, 2, 2};
        float in_data1[] = {1.0f, 1.0f, 1.0f, 1.0f};
        Tensor *input1 = tensor_from_array(in_data1, 4, in_dims1);
        int out_dims1[] = {1, 1, 4, 4};
        Tensor *out1 = tensor_create(4, out_dims1);
        tensor_upsample2d(input1, 2, 2, UPSAMPLE_CUBIC, 0, out1);
        for (size_t i = 0; i < out1->size; i++)
            assert(approx_equal(out1->data[i], 1.0f, EPS));
        tensor_destroy(input1);
        tensor_destroy(out1);
    }
    {
        int in_dims2[] = {1, 1, 3, 3};
        float in_data2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        Tensor *input2 = tensor_from_array(in_data2, 4, in_dims2);
        int out_dims2[] = {1, 1, 3, 3};
        Tensor *out2 = tensor_create(4, out_dims2);
        tensor_upsample2d(input2, 1, 1, UPSAMPLE_CUBIC, 0, out2);
        assert(check_tensor(out2, in_data2, 9));
        tensor_destroy(input2);
        tensor_destroy(out2);
    }
    {
        int in_dims[] = {1, 1, 2, 2};
        float in_data[] = {1, 2, 3, 4};
        Tensor *input = tensor_from_array(in_data, 4, in_dims);
        int out_dims[] = {1, 1, 4, 4};
        Tensor *out = tensor_create(4, out_dims);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_CUBIC, 1, out);
        assert(approx_equal(out->data[0], 1.0f, EPS));
        assert(approx_equal(out->data[3], 2.0f, EPS));
        assert(approx_equal(out->data[12], 3.0f, EPS));
        assert(approx_equal(out->data[15], 4.0f, EPS));
        tensor_destroy(input);
        tensor_destroy(out);
    }
    PASS();
}

void test_conv_transpose1d()
{
    TEST("tensor_conv_transpose1d");
    float in_data[] = {1, 2, 3};
    float w_data[] = {1, 0, -1};
    float bias_val = 0;
    int in_dims[] = {1, 1, 3};
    int w_dims[] = {1, 1, 3};
    int out_dims[] = {1, 1, 5};
    Tensor *input = tensor_from_array(in_data, 3, in_dims);
    Tensor *weight = tensor_from_array(w_data, 3, w_dims);
    Tensor *bias = tensor_wrap(&bias_val, 0, NULL, NULL);
    Tensor *output = tensor_create(3, out_dims);

    ConvParams params = {0};
    params.pad[0] = 0;
    params.stride[0] = 1;
    params.dilation[0] = 1;
    params.groups = 1;

    TensorStatus status = tensor_conv_transpose1d(input, weight, bias, params, output);
    assert(status == TENSOR_OK);
    float expected[] = {1, 2, 2, -2, -3};
    assert(check_tensor(output, expected, 5));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(bias);
    tensor_destroy(output);
    PASS();
}

void test_conv_transpose2d()
{
    TEST("tensor_conv_transpose2d");
    float in_data[] = {1, 2, 3, 4};
    float w_data[] = {1};
    int in_dims[] = {1, 1, 2, 2};
    int w_dims[] = {1, 1, 1, 1};
    int out_dims[] = {1, 1, 2, 2};
    Tensor *input = tensor_from_array(in_data, 4, in_dims);
    Tensor *weight = tensor_from_array(w_data, 4, w_dims);
    Tensor *output = tensor_create(4, out_dims);

    ConvParams params = {0};
    params.pad[0] = 0;
    params.pad[1] = 0;
    params.stride[0] = 1;
    params.stride[1] = 1;
    params.dilation[0] = 1;
    params.dilation[1] = 1;
    params.groups = 1;

    tensor_conv_transpose2d(input, weight, NULL, params, output);
    assert(check_tensor(output, in_data, 4));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_conv_transpose3d()
{
    TEST("tensor_conv_transpose3d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float w_data[] = {1};
    int in_dims[] = {1, 1, 2, 2, 2};
    int w_dims[] = {1, 1, 1, 1, 1};
    int out_dims[] = {1, 1, 2, 2, 2};
    Tensor *input = tensor_from_array(in_data, 5, in_dims);
    Tensor *weight = tensor_from_array(w_data, 5, w_dims);
    Tensor *output = tensor_create(5, out_dims);

    ConvParams params = {0};
    params.pad[0] = 0;
    params.pad[1] = 0;
    params.pad[2] = 0;
    params.stride[0] = 1;
    params.stride[1] = 1;
    params.stride[2] = 1;
    params.dilation[0] = 1;
    params.dilation[1] = 1;
    params.dilation[2] = 1;
    params.groups = 1;

    tensor_conv_transpose3d(input, weight, NULL, params, output);
    assert(check_tensor(output, in_data, 8));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_max_unpool2d()
{
    TEST("tensor_max_unpool2d");
    float x_data[] = {5, 7, 2, 4};
    int x_dims[] = {1, 1, 2, 2};
    float idx_data[] = {0, 5, 10, 15};
    int idx_dims[] = {1, 1, 2, 2};
    int out_size[] = {4, 4};
    int out_dims[] = {1, 1, 4, 4};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *idx = tensor_from_array(idx_data, 4, idx_dims);
    Tensor *out = tensor_create(4, out_dims);

    TensorStatus status = tensor_max_unpool2d(x, idx, out_size, out);
    assert(status == TENSOR_OK);
    float expected[16] = {0};
    expected[0] = 5;
    expected[5] = 7;
    expected[10] = 2;
    expected[15] = 4;
    assert(check_tensor(out, expected, 16));

    tensor_destroy(x);
    tensor_destroy(idx);
    tensor_destroy(out);
    PASS();
}

void test_adaptive_avg_pool2d()
{
    TEST("tensor_adaptive_avg_pool2d");
    float in_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};
    int in_dims[] = {1, 1, 4, 4};
    int out_size[] = {2, 2};
    int out_dims[] = {1, 1, 2, 2};
    Tensor *x = tensor_from_array(in_data, 4, in_dims);
    Tensor *out = tensor_create(4, out_dims);

    tensor_adaptive_avg_pool2d(x, out_size, out);
    float expected[] = {3.5, 5.5, 11.5, 13.5};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

/* ---------- 新增测试 ---------- */

void test_conv2d_groups()
{
    TEST("tensor_conv2d groups>1");
    float in_data[36];
    for (int i = 0; i < 36; i++)
        in_data[i] = i + 1;
    int in_dims[] = {1, 4, 3, 3};
    float w_data[72];
    for (int i = 0; i < 72; i++)
        w_data[i] = 1;
    int w_dims[] = {4, 2, 3, 3};
    int out_dims[] = {1, 4, 1, 1};
    Tensor *input = tensor_from_array(in_data, 4, in_dims);
    Tensor *weight = tensor_from_array(w_data, 4, w_dims);
    Tensor *output = tensor_create(4, out_dims);

    ConvParams params = {0};
    params.pad[0] = 0;
    params.pad[1] = 0;
    params.stride[0] = 1;
    params.stride[1] = 1;
    params.dilation[0] = 1;
    params.dilation[1] = 1;
    params.groups = 2;

    TensorStatus status = tensor_conv2d(input, weight, NULL, params, output);
    assert(status == TENSOR_OK);

    float expected[] = {171, 171, 495, 495};
    assert(check_tensor(output, expected, 4));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_conv2d_params()
{
    TEST("tensor_conv2d with pad/stride/dilation");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int in_dims[] = {1, 1, 3, 3};
    float w_data[] = {1, 1, 1, 1};
    int w_dims[] = {1, 1, 2, 2};
    Tensor *input = tensor_from_array(in_data, 4, in_dims);
    Tensor *weight = tensor_from_array(w_data, 4, w_dims);

    ConvParams params = {0};
    params.pad[0] = 1;
    params.pad[1] = 1;
    params.stride[0] = 2;
    params.stride[1] = 2;
    params.dilation[0] = 1;
    params.dilation[1] = 1;
    params.groups = 1;

    int out_dims[] = {1, 1, 2, 2};
    Tensor *output = tensor_create(4, out_dims);
    TensorStatus status;
    status = tensor_conv2d(input, weight, NULL, params, output);
    assert(status == TENSOR_OK);
    assert(output->dims[2] == 2 && output->dims[3] == 2);
    tensor_destroy(output);

    params.dilation[0] = 2;
    params.dilation[1] = 2;
    params.pad[0] = 1;
    params.pad[1] = 1;
    Tensor *output2 = tensor_create(4, out_dims);
    status = tensor_conv2d(input, weight, NULL, params, output2);
    assert(status == TENSOR_OK);
    tensor_destroy(output2);

    tensor_destroy(input);
    tensor_destroy(weight);
    PASS();
}

void test_conv_transpose1d_params()
{
    TEST("tensor_conv_transpose1d with stride>1, pad>0, dilation>0");
    float in_data[] = {1, 2, 3};
    float w_data[] = {1, 2, 1};
    int in_dims[] = {1, 1, 3};
    int w_dims[] = {1, 1, 3};
    Tensor *input = tensor_from_array(in_data, 3, in_dims);
    Tensor *weight = tensor_from_array(w_data, 3, w_dims);

    ConvParams params = {0};
    params.pad[0] = 1;
    params.stride[0] = 2;
    params.dilation[0] = 1;
    params.groups = 1;

    int out_dims[] = {1, 1, 5};
    Tensor *output = tensor_create(3, out_dims);
    TensorStatus status = tensor_conv_transpose1d(input, weight, NULL, params, output);
    assert(status == TENSOR_OK);
    tensor_destroy(output);

    params.dilation[0] = 2;
    int out_dims2[] = {1, 1, 7};
    Tensor *output2 = tensor_create(3, out_dims2);
    status = tensor_conv_transpose1d(input, weight, NULL, params, output2);
    assert(status == TENSOR_OK);
    tensor_destroy(output2);

    tensor_destroy(input);
    tensor_destroy(weight);
    PASS();
}

void test_pool1d_count_include_pad()
{
    TEST("tensor_pool1d avg with count_include_pad=1");
    float in_data[] = {1, 2, 3, 4};
    int in_dims[] = {1, 1, 4};
    Tensor *input = tensor_from_array(in_data, 3, in_dims);

    PoolParams params = {0};
    params.kernel[0] = 3;
    params.stride[0] = 2;
    params.pad[0] = 1;
    params.ceil_mode = 0;
    params.count_include_pad = 1;

    int out_dims[] = {1, 1, 2};
    Tensor *output = tensor_create(3, out_dims);

    tensor_pool1d(input, POOL_AVG, params, output);
    float expected[] = {1, 3};
    assert(check_tensor(output, expected, 2));

    tensor_destroy(input);
    tensor_destroy(output);
    PASS();
}

void test_lrn_numerical()
{
    TEST("tensor_lrn numerical");
    float in_data[] = {1, 2, 3};
    int in_dims[] = {1, 3, 1, 1};
    Tensor *x = tensor_from_array(in_data, 4, in_dims);
    Tensor *y = tensor_create(4, in_dims);

    int size = 3;
    float alpha = 0.0001f;
    float beta = 0.75f;
    float bias = 1.0f;
    tensor_lrn(x, size, alpha, beta, bias, y);

    float expected[] = {0.999875f, 1.9993f, 2.999025f};
    assert(check_tensor(y, expected, 3));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_gelu_numerical()
{
    TEST("tensor_gelu numerical");
    float x_data[] = {0, 1, -1};
    int dims[] = {3};
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_create(1, dims);
    tensor_gelu(x, y);
    float expected[] = {0.0f, 0.841344746f, -0.158655254f};
    print_tensor(y, "y");
    assert(check_tensor(y, expected, 3));
    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_swish_numerical()
{
    TEST("tensor_swish numerical");
    float x_data[] = {0, 1, -1};
    int dims[] = {3};
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_create(1, dims);
    tensor_swish(x, y);
    float expected[] = {0, 0.7310586f, -0.2689414f};
    assert(check_tensor(y, expected, 3));
    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_mish_numerical()
{
    TEST("tensor_mish numerical");
    float x_data[] = {0, 1, -1};
    int dims[] = {3};
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_create(1, dims);
    tensor_mish(x, y);
    // 更精确的期望值（源自精确数学计算）
    float expected[] = {0.0f, 0.8650984f, -0.30340144f};
    print_tensor(y, "y");
    float diff1 = fabsf(y->data[1] - 0.8650984f);
    float diff2 = fabsf(y->data[2] + 0.30340144f);
    printf("diff1 = %g, diff2 = %g\n", diff1, diff2);
    assert(check_tensor(y, expected, 3));
    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_dropout_statistics()
{
    TEST("tensor_dropout statistics");
    srand(12345);
    int n = 10000;
    int dims[] = {n};
    float p = 0.3f;
    float *x_data = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        x_data[i] = 1.0f;
    Tensor *x = tensor_from_array(x_data, 1, dims);
    Tensor *y = tensor_create(1, dims);
    tensor_dropout(x, p, 1, y);

    int zero_count = 0;
    for (int i = 0; i < n; i++)
    {
        if (y->data[i] == 0)
            zero_count++;
    }
    float observed_p = (float)zero_count / n;
    assert(fabsf(observed_p - p) < 0.02);
    float scale = 1.0f / (1.0f - p);
    for (int i = 0; i < n; i++)
    {
        if (y->data[i] != 0)
        {
            assert(approx_equal(y->data[i], scale, EPS));
        }
    }
    free(x_data);
    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_softmax_axis()
{
    TEST("tensor_softmax axis");
    float data[] = {1, 2, 3, 4, 5, 6};
    int dims[] = {2, 3};
    Tensor *x = tensor_from_array(data, 2, dims);
    Tensor *y = tensor_create(2, dims);

    tensor_softmax(x, 0, y);
    float exp_row0[] = {expf(1), expf(2), expf(3)};
    float exp_row1[] = {expf(4), expf(5), expf(6)};
    float sum_exp_row0 = expf(1) + expf(4);
    float sum_exp_row1 = expf(2) + expf(5);
    float sum_exp_row2 = expf(3) + expf(6);
    float expected_axis0[] = {
        expf(1) / sum_exp_row0, expf(2) / sum_exp_row1, expf(3) / sum_exp_row2,
        expf(4) / sum_exp_row0, expf(5) / sum_exp_row1, expf(6) / sum_exp_row2};
    assert(check_tensor(y, expected_axis0, 6));

    tensor_softmax(x, 1, y);
    float sum_exp0 = expf(1) + expf(2) + expf(3);
    float sum_exp1 = expf(4) + expf(5) + expf(6);
    float expected_axis1[] = {
        expf(1) / sum_exp0, expf(2) / sum_exp0, expf(3) / sum_exp0,
        expf(4) / sum_exp1, expf(5) / sum_exp1, expf(6) / sum_exp1};
    assert(check_tensor(y, expected_axis1, 6));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

/* ---------- 错误测试 ---------- */

void test_conv_errors()
{
    TEST("convolution error paths");
    int in_dims[] = {1, 1, 5};
    int w_dims[] = {1, 1, 3};
    float dummy[10];
    Tensor *input = tensor_from_array(dummy, 3, in_dims);
    Tensor *weight = tensor_from_array(dummy, 3, w_dims);
    Tensor *output = tensor_create(3, (int[]){1, 1, 3});

    ConvParams params = {0};
    params.pad[0] = 0;
    params.stride[0] = 1;
    params.dilation[0] = 1;
    params.groups = 1;

    Tensor *input4d = tensor_create(4, (int[]){1, 1, 5, 1});
    TensorStatus st = tensor_conv1d(input4d, weight, NULL, params, output);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(input4d);

    Tensor *weight_bad = tensor_from_array(dummy, 3, (int[]){2, 1, 3});
    st = tensor_conv1d(input, weight_bad, NULL, params, output);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(weight_bad);

    params.stride[0] = 0;
    st = tensor_conv1d(input, weight, NULL, params, output);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    params.stride[0] = 1;

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_norm_errors()
{
    TEST("normalization error paths");
    int x_dims[] = {2, 2, 1, 1};
    float x_data[4] = {1, 2, 3, 4};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, x_dims);

    Tensor *mean_bad = tensor_create(1, (int[]){3});
    Tensor *var_bad = tensor_create(1, (int[]){3});
    TensorStatus st = tensor_batchnorm(x, mean_bad, var_bad, NULL, NULL, 1e-5, y);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(mean_bad);
    tensor_destroy(var_bad);

    Tensor *scale_bad = tensor_create(1, (int[]){4});
    st = tensor_layernorm(x, scale_bad, NULL, 1e-5, y);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(scale_bad);

    st = tensor_instancenorm(x, scale_bad, NULL, 1e-5, y);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    st = tensor_groupnorm(x, NULL, NULL, 3, 1e-5, y);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_linear_error()
{
    TEST("tensor_linear error");
    int in_dims[] = {2, 3};
    float in_data[6] = {0};
    int w_dims[] = {4, 2};
    Tensor *input = tensor_from_array(in_data, 2, in_dims);
    Tensor *weight = tensor_create(2, w_dims);
    Tensor *output = tensor_create(2, (int[]){2, 4});
    TensorStatus st = tensor_linear(input, weight, NULL, output);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_max_unpool_error()
{
    TEST("tensor_max_unpool2d index out of bounds");
    float x_data[] = {5};
    int x_dims[] = {1, 1, 1, 1};
    float idx_data[] = {100};
    int idx_dims[] = {1, 1, 1, 1};
    int out_size[] = {4, 4};
    int out_dims[] = {1, 1, 4, 4};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *idx = tensor_from_array(idx_data, 4, idx_dims);
    Tensor *out = tensor_create(4, out_dims);
    TensorStatus st = tensor_max_unpool2d(x, idx, out_size, out);
    assert(st == TENSOR_ERR_INDEX_OUT_OF_BOUNDS);
    tensor_destroy(x);
    tensor_destroy(idx);
    tensor_destroy(out);
    PASS();
}

void test_adaptive_pool_error()
{
    TEST("tensor_adaptive_avg_pool2d output_size > input");
    int in_dims[] = {1, 1, 2, 2};
    float in_data[4] = {0};
    int out_size[] = {3, 3};
    int out_dims[] = {1, 1, 3, 3};
    Tensor *x = tensor_from_array(in_data, 4, in_dims);
    Tensor *out = tensor_create(4, out_dims);
    TensorStatus st = tensor_adaptive_avg_pool2d(x, out_size, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_embedding()
{
    TEST("tensor_embedding");
    float in_data[] = {0, 1, 2, 0};
    int in_dims[] = {2, 2};
    float w_data[] = {1, 2, 3, 4, 5, 6}; // vocab_size=3, emb_dim=2
    int w_dims[] = {3, 2};
    int out_dims[] = {2, 2, 2};
    Tensor *input = tensor_from_array(in_data, 2, in_dims);
    Tensor *weight = tensor_from_array(w_data, 2, w_dims);
    Tensor *out = tensor_create(3, out_dims);

    // 无 padding_idx
    tensor_embedding(input, weight, -1, out);
    float expected_no_pad[] = {1, 2, 3, 4, 5, 6, 1, 2}; // 已验证
    assert(check_tensor(out, expected_no_pad, 8));

    // padding_idx = 0
    tensor_embedding(input, weight, 0, out);
    float expected_pad[] = {0, 0, 3, 4, 5, 6, 0, 0};
    assert(check_tensor(out, expected_pad, 8));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(out);
    PASS();
}

void test_upsample1d()
{
    TEST("tensor_upsample1d");
    float in_data[] = {1, 2, 3, 4};
    int in_dims[] = {1, 2, 2}; // N=1, C=2, L=2
    int out_dims[] = {1, 2, 4};
    Tensor *x = tensor_from_array(in_data, 3, in_dims);
    Tensor *out = tensor_create(3, out_dims);

    // 最近邻
    tensor_upsample1d(x, 2, UPSAMPLE_NEAREST, 0, out);
    float expected_nearest[] = {1, 1, 2, 2, 3, 3, 4, 4};
    assert(check_tensor(out, expected_nearest, 8));

    // 线性插值 align_corners=0
    tensor_upsample1d(x, 2, UPSAMPLE_LINEAR, 0, out);
    float expected_linear0[] = {1, 1.25f, 1.75f, 2, 3, 3.25f, 3.75f, 4};
    assert(check_tensor(out, expected_linear0, 8));

    // 线性插值 align_corners=1
    tensor_upsample1d(x, 2, UPSAMPLE_LINEAR, 1, out);
    float expected_linear1[] = {1, 4.0f / 3, 5.0f / 3, 2, 3, 10.0f / 3, 11.0f / 3, 4};
    assert(check_tensor(out, expected_linear1, 8));

    // 三次插值 align_corners=0
    tensor_upsample1d(x, 2, UPSAMPLE_CUBIC, 0, out);
    float expected_cubic0[] = {1.15625, 1.15625, 1.84375, 1.84375, 3.15625, 3.15625, 3.84375, 3.84375}; // 占位
    assert(check_tensor(out, expected_cubic0, 8));

    // 三次插值 align_corners=1
    tensor_upsample1d(x, 2, UPSAMPLE_CUBIC, 1, out);
    float expected_cubic1[] = {1.0, 1.2592592, 1.7407408, 2.0, 3.0, 3.2592592, 3.7407408, 4.0};
    assert(check_tensor(out, expected_cubic1, 8));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_upsample3d()
{
    TEST("tensor_upsample3d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8}; // N=1,C=1,D=2,H=2,W=2
    int in_dims[] = {1, 1, 2, 2, 2};
    int out_dims[] = {1, 1, 4, 4, 4};
    Tensor *x = tensor_from_array(in_data, 5, in_dims);
    Tensor *out = tensor_create(5, out_dims);

    // 最近邻
    tensor_upsample3d(x, 2, 2, 2, UPSAMPLE_NEAREST, 0, out);
    float expected_nearest[] = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 7.0, 7.0, 8.0, 8.0, 5.0, 5.0, 6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 7.0, 7.0, 8.0, 8.0};
    assert(check_tensor(out, expected_nearest, 64));

    // 线性插值 align_corners=0
    tensor_upsample3d(x, 2, 2, 2, UPSAMPLE_LINEAR, 0, out);
    float expected_linear0[] = {1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0, 2.0, 2.25, 2.75, 3.0, 2.5, 2.75, 3.25, 3.5, 3.5, 3.75, 4.25, 4.5, 4.0, 4.25, 4.75, 5.0, 4.0, 4.25, 4.75, 5.0, 4.5, 4.75, 5.25, 5.5, 5.5, 5.75, 6.25, 6.5, 6.0, 6.25, 6.75, 7.0, 5.0, 5.25, 5.75, 6.0, 5.5, 5.75, 6.25, 6.5, 6.5, 6.75, 7.25, 7.5, 7.0, 7.25, 7.75, 8.0};
    assert(check_tensor(out, expected_linear0, 64));

    // 线性插值 align_corners=1
    tensor_upsample3d(x, 2, 2, 2, UPSAMPLE_LINEAR, 1, out);
    float expected_linear1[] = {1.0, 1.3333333, 1.6666667, 2.0, 1.6666666, 2.0, 2.3333335, 2.6666665, 2.3333333, 2.6666667, 3.0000002, 3.3333335, 3.0, 3.3333333, 3.6666667, 4.0, 2.3333335, 2.6666665, 3.0, 3.3333333, 3.0, 3.333333, 3.6666667, 4.0, 3.6666665, 4.0, 4.3333335, 4.666667, 4.3333335, 4.6666665, 5.0, 5.333333, 3.6666667, 3.9999998, 4.333333, 4.6666665, 4.333333, 4.6666665, 5.0, 5.3333335, 5.0, 5.3333335, 5.666667, 6.0000005, 5.666667, 6.0, 6.333334, 6.666667, 5.0, 5.333333, 5.6666665, 6.0, 5.6666665, 5.9999995, 6.333333, 6.6666665, 6.3333335, 6.6666665, 7.0000005, 7.3333335, 7.0, 7.333333, 7.666667, 8.0};
    assert(check_tensor(out, expected_linear1, 64));

    // 三次插值 align_corners=0
    tensor_upsample3d(x, 2, 2, 2, UPSAMPLE_CUBIC, 0, out);
    float expected_cubic0[] = {2.09375, 2.09375, 2.78125, 2.78125, 2.09375, 2.09375, 2.78125, 2.78125, 3.46875, 3.46875, 4.15625, 4.15625, 3.46875, 3.46875, 4.15625, 4.15625, 2.09375, 2.09375, 2.78125, 2.78125, 2.09375, 2.09375, 2.78125, 2.78125, 3.46875, 3.46875, 4.15625, 4.15625, 3.46875, 3.46875, 4.15625, 4.15625, 4.84375, 4.84375, 5.53125, 5.53125, 4.84375, 4.84375, 5.53125, 5.53125, 6.21875, 6.21875, 6.90625, 6.90625, 6.21875, 6.21875, 6.90625, 6.90625, 4.84375, 4.84375, 5.53125, 5.53125, 4.84375, 4.84375, 5.53125, 5.53125, 6.21875, 6.21875, 6.90625, 6.90625, 6.21875, 6.21875, 6.90625, 6.90625};
    print_tensor(out,"cubic0");
    assert(check_tensor(out, expected_cubic0, 64));

    // 三次插值 align_corners=1
    tensor_upsample3d(x, 2, 2, 2, UPSAMPLE_CUBIC, 1, out);
    float expected_cubic1[] = {1.0, 1.2592592, 1.7407408, 2.0, 1.5185186, 1.7777778, 2.2592592, 2.5185184, 2.4814816, 2.7407408, 3.2222223, 3.4814816, 3.0, 3.2592592, 3.7407408, 4.0, 2.0370371, 2.2962964, 2.7777777, 3.0370371, 2.5555556, 2.8148148, 3.2962964, 3.5555556, 3.5185184, 3.7777777, 4.259259, 4.5185184, 4.037037, 4.296296, 4.7777777, 5.037037, 3.9629629, 4.2222223, 4.703704, 4.962963, 4.4814816, 4.740741, 5.2222223, 5.4814816, 5.4444447, 5.703704, 6.185185, 6.4444447, 5.962963, 6.2222223, 6.703704, 6.962963, 5.0, 5.259259, 5.740741, 6.0, 5.5185184, 5.7777777, 6.259259, 6.5185184, 6.4814816, 6.740741, 7.2222223, 7.4814816, 7.0, 7.259259, 7.740741, 8.0};
    print_tensor(out,"cubic1");
    assert(check_tensor(out, expected_cubic1, 64));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_adaptive_avg_pool1d()
{
    TEST("tensor_adaptive_avg_pool1d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int in_dims[] = {2, 1, 4}; // N=2, C=1, L=4
    int out_size[] = {2};
    int out_dims[] = {2, 1, 2};
    Tensor *x = tensor_from_array(in_data, 3, in_dims);
    Tensor *out = tensor_create(3, out_dims);

    tensor_adaptive_avg_pool1d(x, out_size, out);
    float expected[] = {1.5, 3.5, 5.5, 7.5}; // 已验证
    assert(check_tensor(out, expected, 4));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_adaptive_avg_pool3d()
{
    TEST("tensor_adaptive_avg_pool3d");
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // N=1,C=1,D=2,H=2,W=4
    int in_dims[] = {1, 1, 2, 2, 4};
    int out_size[] = {1, 1, 2};
    int out_dims[] = {1, 1, 1, 1, 2};
    Tensor *x = tensor_from_array(in_data, 5, in_dims);
    Tensor *out = tensor_create(5, out_dims);

    tensor_adaptive_avg_pool3d(x, out_size, out);
    float expected[] = {7.5, 9.5};
    assert(check_tensor(out, expected, 2));

    tensor_destroy(x);
    tensor_destroy(out);
    PASS();
}

void test_max_unpool1d()
{
    TEST("tensor_max_unpool1d");
    float x_data[] = {5, 7};
    int x_dims[] = {1, 1, 2};
    float idx_data[] = {2, 5}; // 输出长度为6
    int idx_dims[] = {1, 1, 2};
    int out_size[] = {6};
    int out_dims[] = {1, 1, 6};
    Tensor *x = tensor_from_array(x_data, 3, x_dims);
    Tensor *idx = tensor_from_array(idx_data, 3, idx_dims);
    Tensor *out = tensor_create(3, out_dims);

    tensor_max_unpool1d(x, idx, out_size, out);
    float expected[6] = {0, 0, 5, 0, 0, 7};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(x);
    tensor_destroy(idx);
    tensor_destroy(out);
    PASS();
}

void test_max_unpool3d()
{
    TEST("tensor_max_unpool3d");
    float x_data[] = {5, 7};
    int x_dims[] = {1, 1, 1, 1, 2}; // N=1,C=1,D=1,H=1,W=2
    float idx_data[] = {10, 35};    // 输出尺寸 3x3x4 = 36
    int idx_dims[] = {1, 1, 1, 1, 2};
    int out_size[] = {3, 3, 4};
    int out_dims[] = {1, 1, 3, 3, 4};
    Tensor *x = tensor_from_array(x_data, 5, x_dims);
    Tensor *idx = tensor_from_array(idx_data, 5, idx_dims);
    Tensor *out = tensor_create(5, out_dims);

    tensor_max_unpool3d(x, idx, out_size, out);
    // 验证索引10和35位置的值
    float *data = out->data;
    assert(approx_equal(data[10], 5, EPS));
    assert(approx_equal(data[35], 7, EPS));

    tensor_destroy(x);
    tensor_destroy(idx);
    tensor_destroy(out);
    PASS();
}

/* ---------- 主函数 ---------- */
int main()
{
    test_conv1d();
    test_conv2d();
    test_conv3d();
    test_pool1d();
    test_pool2d();
    test_pool3d();
    test_global_pool2d();
    test_batchnorm();
    test_layernorm();
    test_instancenorm();
    test_groupnorm();
    test_lrn();
    test_activations();
    test_prelu();
    test_linear();
    test_dropout();
    test_softmax();
    test_upsample2d_nearest();
    test_upsample2d_linear();
    test_upsample2d_cubic();

    test_conv_transpose1d();
    test_conv_transpose2d();
    test_conv_transpose3d();

    test_max_unpool2d();
    test_adaptive_avg_pool2d();

    test_conv2d_groups();
    test_conv2d_params();
    test_conv_transpose1d_params();
    test_pool1d_count_include_pad();
    test_lrn_numerical();
    test_gelu_numerical();
    test_swish_numerical();
    test_mish_numerical();
    test_dropout_statistics();
    test_softmax_axis();
    test_conv_errors();
    test_norm_errors();
    test_linear_error();
    test_max_unpool_error();
    test_adaptive_pool_error();

    test_embedding();
    test_upsample1d();
    test_upsample3d();
    test_adaptive_avg_pool1d();
    test_adaptive_avg_pool3d();
    test_max_unpool1d();
    test_max_unpool3d();
    printf("All nn_ops tests passed!\n");
    return 0;
}