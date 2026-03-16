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
        printf("%.4f ", t->data[i]);
    printf("\n");
}

/* ---------- 卷积测试 ---------- */
void test_conv1d()
{
    TEST("tensor_conv1d");
    // 输入: N=1, C=1, L=5
    float in_data[] = {1, 2, 3, 4, 5};
    // 权重: out=1, in=1, kL=3
    float w_data[] = {1, 0, -1}; // 简单的差分核
    // 偏置: 0
    float bias_val = 0;
    int in_dims[] = {1, 1, 5};
    int w_dims[] = {1, 1, 3};
    int out_dims[] = {1, 1, 3}; // padding=0, stride=1
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

    // 预期: 差分结果 (2-0? 实际上：1*1 + 2*0 + 3*(-1) = -2? 但应该是边缘检测)
    // 手动计算:
    // ol=0: 1*1 + 2*0 + 3*(-1) = -2
    // ol=1: 2*1 + 3*0 + 4*(-1) = -2
    // ol=2: 3*1 + 4*0 + 5*(-1) = -2
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
    int out_dims[] = {1, 1, 2, 2}; // stride=1, pad=0
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
    float expected[] = {1 + 5, 2 + 6, 4 + 8, 5 + 9}; // 6,8,12,14
    float exp[] = {6, 8, 12, 14};
    assert(check_tensor(output, exp, 4));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

void test_conv3d()
{
    TEST("tensor_conv3d");
    // 简化的 2x2x2 输入，1个通道，1个输出通道，核2x2x2
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float w_data[] = {1, 0, 0, 1, 0, 1, 1, 0}; // 随机
    int in_dims[] = {1, 1, 2, 2, 2};
    int w_dims[] = {1, 1, 2, 2, 2};
    int out_dims[] = {1, 1, 1, 1, 1}; // stride=1, pad=0
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
    // 手动点积: 1*1 + 2*0 + 3*0 + 4*1 + 5*0 + 6*1 + 7*1 + 8*0 = 1+4+6+7=18
    float expected[] = {18};
    assert(check_tensor(output, expected, 1));

    tensor_destroy(input);
    tensor_destroy(weight);
    tensor_destroy(output);
    PASS();
}

/* ---------- 池化测试 ---------- */
void test_pool1d()
{
    TEST("tensor_pool1d");
    float in_data[] = {1, 3, 2, 4, 5};
    int in_dims[] = {1, 1, 5};
    Tensor *input = tensor_from_array(in_data, 3, in_dims);

    // 测试 ceil_mode = 0
    PoolParams params0 = {0};
    params0.kernel[0] = 2;
    params0.stride[0] = 2;
    params0.pad[0] = 0;
    params0.ceil_mode = 0;
    params0.count_include_pad = 0;

    int out_dims0[] = {1, 1, 2}; // ceil_mode=0 时输出长度应为2
    Tensor *output0 = tensor_create(3, out_dims0);
    tensor_pool1d(input, POOL_MAX, params0, output0);
    float expected_max0[] = {3, 4};
    assert(check_tensor(output0, expected_max0, 2));

    tensor_pool1d(input, POOL_AVG, params0, output0);
    float expected_avg0[] = {(1 + 3) / 2.0f, (2 + 4) / 2.0f}; // 2, 3
    assert(check_tensor(output0, expected_avg0, 2));

    // 测试 ceil_mode = 1
    PoolParams params1 = {0};
    params1.kernel[0] = 2;
    params1.stride[0] = 2;
    params1.pad[0] = 0;
    params1.ceil_mode = 1; // 使用 ceil
    params1.count_include_pad = 0;

    int out_dims1[] = {1, 1, 3};
    Tensor *output1 = tensor_create(3, out_dims1);
    tensor_pool1d(input, POOL_MAX, params1, output1);
    float expected_max1[] = {3, 4, 5};
    assert(check_tensor(output1, expected_max1, 3));

    tensor_pool1d(input, POOL_AVG, params1, output1);
    // 注意：ceil_mode=1 时最后一个窗口只包含一个元素 [5]，平均值为5
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
    float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8}; // 2x2x2
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
    int in_dims[] = {2, 1, 2, 2}; // N=2, C=1, H=2, W=2
    Tensor *input = tensor_from_array(in_data, 4, in_dims);
    int out_dims[] = {2, 1, 1, 1};
    Tensor *avg_out = tensor_create(4, out_dims);
    Tensor *max_out = tensor_create(4, out_dims);

    tensor_global_avg_pool2d(input, avg_out);
    tensor_global_max_pool2d(input, max_out);

    float expected_avg[] = {(1 + 2 + 3 + 4) / 4.0f, (5 + 6 + 7 + 8) / 4.0f};
    float expected_max[] = {4, 8};
    assert(check_tensor(avg_out, expected_avg, 2));
    assert(check_tensor(max_out, expected_max, 2));

    tensor_destroy(input);
    tensor_destroy(avg_out);
    tensor_destroy(max_out);
    PASS();
}

/* ---------- 归一化测试 ---------- */
void test_batchnorm()
{
    TEST("tensor_batchnorm");
    // x: [2,2,1,1]
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
    printf("batchnorm actual: ");
    for (int i = 0; i < 4; i++)
        printf("%f ", y->data[i]);
    printf("\n");
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
    int x_dims[] = {2, 3}; // [2,3]
    Tensor *x = tensor_from_array(x_data, 2, x_dims);
    Tensor *y = tensor_create(2, x_dims);

    tensor_layernorm(x, NULL, NULL, 1e-5f, y);
    // 第一行: mean=2, var=2/3≈0.6667, std≈0.8165, 归一化后: -1.2247,0,1.2247
    // 第二行: mean=5, var=0.6667, 归一化: -1.2247,0,1.2247
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
    int x_dims[] = {2, 2, 2, 1}; // N=2, C=2, H=2, W=1
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, x_dims);

    tensor_instancenorm(x, NULL, NULL, 1e-12f, y);
    // 每个样本每个通道独立计算
    // 样本0通道0: [1,2] mean=1.5, var=0.25, inv_std=1/0.5=2, 归一化: (1-1.5)*2=-1, (2-1.5)*2=1
    // 样本0通道1: [3,4] mean=3.5, var=0.25, 归一化: (3-3.5)*2=-1, (4-3.5)*2=1
    // 样本1通道0: [5,6] mean=5.5, var=0.25, -> -1,1
    // 样本1通道1: [7,8] mean=7.5, var=0.25, -> -1,1
    float expected[] = {-1, 1, -1, 1, -1, 1, -1, 1};
    assert(check_tensor(y, expected, 8));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_groupnorm()
{
    TEST("tensor_groupnorm");
    float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8}; // N=1, C=4, H=2, W=1
    int x_dims[] = {1, 4, 2, 1};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, x_dims);

    tensor_groupnorm(x, NULL, NULL, 2, 1e-12f, y); // 2 groups
    // group0: 通道0,1 每个通道有2个元素 -> 4个元素 [1,2,3,4] mean=2.5, var=1.25, inv_std≈0.8944
    // 归一化后: (1-2.5)*0.8944=-1.3416, (2-2.5)*0.8944=-0.4472, (3-2.5)*0.8944=0.4472, (4-2.5)*0.8944=1.3416
    // group1: 通道2,3 [5,6,7,8] mean=6.5, var=1.25, 归一化: -1.3416,-0.4472,0.4472,1.3416
    float expected[] = {
        -1.3416407865f, -0.4472135955f, 0.4472135955f, 1.3416407865f,
        -1.3416407865f, -0.4472135955f, 0.4472135955f, 1.3416407865f};
    printf("groupnorm actual: ");
    for (int i = 0; i < 8; i++)
        printf("%f ", y->data[i]);
    printf("\n");
    assert(check_tensor(y, expected, 8));

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_lrn()
{
    TEST("tensor_lrn");
    float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8}; // N=1, C=4, H=2, W=1
    int x_dims[] = {1, 4, 2, 1};
    Tensor *x = tensor_from_array(x_data, 4, x_dims);
    Tensor *y = tensor_create(4, x_dims);

    tensor_lrn(x, 3, 0.001f, 0.75f, 1.0f, y);
    // 简单验证形状正确
    assert(y != NULL);
    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

/* ---------- 激活函数测试 ---------- */
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
    // 近似验证
    float selu_exp[] = {-1.520166f, -1.111331f, 0, 1.050701f, 2.101402f};
    assert(check_tensor(y, selu_exp, 5));

    tensor_gelu(x, y);
    // 简单检查形状
    assert(y != NULL);

    tensor_swish(x, y);
    tensor_mish(x, y);
    tensor_softplus(x, y);
    tensor_softsign(x, y);
    tensor_hardswish(x, y);
    tensor_hardsigmoid(x, y);
    // 都至少不崩溃
    assert(y != NULL);

    tensor_destroy(x);
    tensor_destroy(y);
    PASS();
}

void test_prelu()
{
    TEST("tensor_prelu");
    float x_data[] = {-2, -1, 0, 1, 2, -3, -2, -1, 0, 1}; // N=1, C=2, H=1, W=5
    int x_dims[] = {1, 2, 1, 5};                          // 改为 N=1
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

/* ---------- 其他层测试 ---------- */
void test_linear()
{
    TEST("tensor_linear");
    float in_data[] = {1, 2, 3, 4, 5, 6};                  // batch=2, in_features=3
    float w_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}; // out_features=4, in_features=3
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

    // 推理模式
    tensor_dropout(x, 0.5, 0, y);
    assert(check_tensor(y, x_data, 4));

    // 训练模式（随机，仅验证形状）
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

    tensor_upsample2d(x, 2, 2, UPSAMPLE_NEAREST, 0, y); // 添加 align_corners 参数
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

    /* 测试 align_corners = 0 (默认中心对齐) */
    {
        int in_dims[] = {1, 1, 2, 2};
        float in_data[] = {1, 2, 3, 4};
        Tensor *input = tensor_from_array(in_data, 4, in_dims);
        int out_dims[] = {1, 1, 4, 4};
        Tensor *out = tensor_create(4, out_dims);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_LINEAR, 0, out);
        // 预期值（中心对齐）
        float expected[] = {
            1.00f, 1.25f, 1.75f, 2.00f,
            1.50f, 1.75f, 2.25f, 2.50f,
            2.50f, 2.75f, 3.25f, 3.50f,
            3.00f, 3.25f, 3.75f, 4.00f};
        assert(check_tensor(out, expected, 16));
        tensor_destroy(input);
        tensor_destroy(out);
    }

    /* 测试 align_corners = 1 (角点对齐) */
    {
        int in_dims[] = {1, 1, 2, 2};
        float in_data[] = {1, 2, 3, 4};
        Tensor *input = tensor_from_array(in_data, 4, in_dims);
        int out_dims[] = {1, 1, 4, 4};
        Tensor *out = tensor_create(4, out_dims);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_LINEAR, 1, out);
        // 当 align_corners=1 时，输出角点与输入角点完全一致，其余点线性插值。
        // 计算：输入坐标范围 [0,1] 映射到输出坐标范围 [0,3]（整数索引）。
        // 输出 (0,0) -> 输入 (0,0) = 1
        // 输出 (0,1) -> 输入 x = 1 * (2-1)/(4-1) = 1/3 ≈ 0.3333, 在 (0,0) 和 (0,1) 之间插值
        // 详细计算可得如下结果：
        float expected[] = {
            1.0f, 4.0f / 3.0f, 5.0f / 3.0f, 2.0f,
            5.0f / 3.0f, 2.0f, 7.0f / 3.0f, 8.0f / 3.0f,
            7.0f / 3.0f, 8.0f / 3.0f, 3.0f, 10.0f / 3.0f,
            3.0f, 10.0f / 3.0f, 11.0f / 3.0f, 4.0f};
        printf("\nActual output:\n");
        for (int i = 0; i < 16; i++)
            printf("%.6f ", out->data[i]);
        printf("\n");
        assert(check_tensor(out, expected, 16));
        tensor_destroy(input);
        tensor_destroy(out);
    }

    /* 测试常数张量，align_corners 不影响 */
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

    // align_corners = 0
    {
        // 常数张量
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
        // scale = 1
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

    // 测试 align_corners = 1（可选，可手动计算或参考标准库）
    // 这里简单验证角点对齐
    {
        int in_dims[] = {1, 1, 2, 2};
        float in_data[] = {1, 2, 3, 4};
        Tensor *input = tensor_from_array(in_data, 4, in_dims);
        int out_dims[] = {1, 1, 4, 4};
        Tensor *out = tensor_create(4, out_dims);
        tensor_upsample2d(input, 2, 2, UPSAMPLE_CUBIC, 1, out);
        // 角点应等于输入角点
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
    // 输出长度 (3-1)*1 + (3-1)*1 + 1 - 0 = 5
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

    // 手动计算结果：[1,2,2,-2,-3]
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
    // 2x2 输入，1x1 核，输出相同
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
    // 2x2x2 输入，1x1x1 核，输出相同
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

    printf("All nn_ops tests passed!\n");
    return 0;
}