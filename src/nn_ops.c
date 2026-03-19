#include "tensor.h"
#include "nn_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/**
 * @file nn_ops.c
 * @brief 神经网络层操作的实现：卷积、池化、归一化、激活函数等
 *
 * 包含一维/二维/三维卷积及转置卷积、各类池化、批归一化/层归一化/实例归一化/组归一化/LRN、
 * 多种激活函数、全连接层、dropout、softmax/log_softmax、上采样（最近邻/线性/三次插值）、
 * 最大反池化、自适应平均池化。
 */

/* ==================== 内部辅助函数 ==================== */

/**
 * @brief 反射坐标（用于三次插值的边界处理）
 * @param coord 原始坐标
 * @param dim   维度大小
 * @return 反射后的有效坐标
 */
static int reflect_coord(int coord, int dim)
{
    if (dim <= 1)
        return 0;
    int period = 2 * dim - 2;
    coord %= period;
    if (coord < 0)
        coord += period;
    return (coord < dim) ? coord : period - coord;
}

/**
 * @brief Catmull-Rom 插值（三次插值核心）
 * @param p0, p1, p2, p3 四个控制点
 * @param t 插值位置 [0,1]
 * @return 插值结果
 */
static float catmull_rom(float p0, float p1, float p2, float p3, float t)
{
    return 0.5f * ((-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t +
                   (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                   (-p0 + p2) * t +
                   2 * p1);
}

/* ---------- 卷积输出形状计算 ---------- */

static void conv1d_output_shape(int in_len, int kernel_len,
                                int pad, int stride, int dilation,
                                int *out_len)
{
    int effective_kernel = (kernel_len - 1) * dilation + 1;
    *out_len = (in_len + 2 * pad - effective_kernel) / stride + 1;
}

static void conv2d_output_shape(int in_h, int in_w,
                                int kernel_h, int kernel_w,
                                int pad_h, int pad_w,
                                int stride_h, int stride_w,
                                int dilation_h, int dilation_w,
                                int *out_h, int *out_w)
{
    int effective_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int effective_kernel_w = (kernel_w - 1) * dilation_w + 1;
    *out_h = (in_h + 2 * pad_h - effective_kernel_h) / stride_h + 1;
    *out_w = (in_w + 2 * pad_w - effective_kernel_w) / stride_w + 1;
}

static void conv3d_output_shape(int in_d, int in_h, int in_w,
                                int kernel_d, int kernel_h, int kernel_w,
                                int pad_d, int pad_h, int pad_w,
                                int stride_d, int stride_h, int stride_w,
                                int dilation_d, int dilation_h, int dilation_w,
                                int *out_d, int *out_h, int *out_w)
{
    int effective_kernel_d = (kernel_d - 1) * dilation_d + 1;
    int effective_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int effective_kernel_w = (kernel_w - 1) * dilation_w + 1;
    *out_d = (in_d + 2 * pad_d - effective_kernel_d) / stride_d + 1;
    *out_h = (in_h + 2 * pad_h - effective_kernel_h) / stride_h + 1;
    *out_w = (in_w + 2 * pad_w - effective_kernel_w) / stride_w + 1;
}

/* ---------- 转置卷积输出形状计算 ---------- */

static void conv_transpose1d_output_shape(int in_len, int kernel_len,
                                          int pad, int stride, int dilation,
                                          int *out_len)
{
    *out_len = (in_len - 1) * stride + (kernel_len - 1) * dilation + 1 - 2 * pad;
}

static void conv_transpose2d_output_shape(int in_h, int in_w,
                                          int kernel_h, int kernel_w,
                                          int pad_h, int pad_w,
                                          int stride_h, int stride_w,
                                          int dilation_h, int dilation_w,
                                          int *out_h, int *out_w)
{
    *out_h = (in_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - 2 * pad_h;
    *out_w = (in_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - 2 * pad_w;
}

static void conv_transpose3d_output_shape(int in_d, int in_h, int in_w,
                                          int kernel_d, int kernel_h, int kernel_w,
                                          int pad_d, int pad_h, int pad_w,
                                          int stride_d, int stride_h, int stride_w,
                                          int dilation_d, int dilation_h, int dilation_w,
                                          int *out_d, int *out_h, int *out_w)
{
    *out_d = (in_d - 1) * stride_d + (kernel_d - 1) * dilation_d + 1 - 2 * pad_d;
    *out_h = (in_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - 2 * pad_h;
    *out_w = (in_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - 2 * pad_w;
}

/* ==================== 卷积层 ==================== */

/**
 * @brief 一维卷积
 * @param input  输入张量 [N, C, L]
 * @param weight 卷积核 [out_channels, in_channels/groups, kL]
 * @param bias   偏置 [out_channels] 或标量，可为 NULL
 * @param params 卷积参数（pad[0], stride[0], dilation[0], groups）
 * @param output 输出张量 [N, out_channels, out_len]
 * @return TensorStatus
 */
TensorStatus tensor_conv1d(const Tensor *input, const Tensor *weight,
                           const Tensor *bias, ConvParams params,
                           Tensor *output)
{
    if (!input || !weight || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 3 || weight->ndim != 3)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (params.pad[0] < 0 || params.stride[0] <= 0 || params.dilation[0] <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (params.groups <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    int N = input->dims[0];
    int C = input->dims[1];
    int L = input->dims[2];
    int out_channels = weight->dims[0];
    int in_channels = weight->dims[1];
    int kL = weight->dims[2];
    int groups = params.groups;

    if (C != in_channels * groups)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (out_channels % groups != 0)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int group_in_channels = in_channels;
    int group_out_channels = out_channels / groups;

    int out_len;
    conv1d_output_shape(L, kL, params.pad[0], params.stride[0], params.dilation[0], &out_len);
    if (out_len <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (output->ndim != 3 ||
        output->dims[0] != N ||
        output->dims[1] != out_channels ||
        output->dims[2] != out_len)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[3], w_strides[3], out_strides[3];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(output, out_strides);

    util_clear_tensor(output);

    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < groups; g++)
        {
            for (int oc = 0; oc < group_out_channels; oc++)
            {
                int out_c = g * group_out_channels + oc;
                for (int ol = 0; ol < out_len; ol++)
                {
                    float sum = 0.0f;
                    int in_l0 = ol * params.stride[0] - params.pad[0];
                    for (int ic = 0; ic < group_in_channels; ic++)
                    {
                        int in_c = g * group_in_channels + ic;
                        for (int kl = 0; kl < kL; kl++)
                        {
                            int in_l = in_l0 + kl * params.dilation[0];
                            if (in_l < 0 || in_l >= L)
                                continue;
                            size_t in_off = n * in_strides[0] +
                                            in_c * in_strides[1] +
                                            in_l * in_strides[2];
                            size_t w_off = out_c * w_strides[0] +
                                           ic * w_strides[1] +
                                           kl * w_strides[2];
                            sum += input->data[in_off] * weight->data[w_off];
                        }
                    }
                    if (bias)
                    {
                        if (bias->ndim == 1 && (int)bias->size == out_channels)
                            sum += bias->data[out_c];
                        else if (bias->ndim == 0)
                            sum += bias->data[0];
                        else
                            return TENSOR_ERR_SHAPE_MISMATCH;
                    }
                    size_t out_off = n * out_strides[0] +
                                     out_c * out_strides[1] +
                                     ol * out_strides[2];
                    output->data[out_off] = sum;
                }
            }
        }
    }
    return TENSOR_OK;
}

/**
 * @brief 二维卷积
 * @param input  输入张量 [N, C, H, W]
 * @param weight 卷积核 [out_channels, in_channels/groups, kH, kW]
 * @param bias   偏置 [out_channels] 或标量，可为 NULL
 * @param params 卷积参数（pad[0..1], stride[0..1], dilation[0..1], groups）
 * @param output 输出张量 [N, out_channels, out_h, out_w]
 * @return TensorStatus
 */
TensorStatus tensor_conv2d(const Tensor *input, const Tensor *weight,
                           const Tensor *bias, ConvParams params,
                           Tensor *output)
{
    if (!input || !weight || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 4 || weight->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (params.pad[0] < 0 || params.pad[1] < 0 ||
        params.stride[0] <= 0 || params.stride[1] <= 0 ||
        params.dilation[0] <= 0 || params.dilation[1] <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (params.groups <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    int N = input->dims[0];
    int C = input->dims[1];
    int H = input->dims[2];
    int W = input->dims[3];
    int out_channels = weight->dims[0];
    int in_channels = weight->dims[1];
    int kH = weight->dims[2];
    int kW = weight->dims[3];
    int groups = params.groups;

    if (C != in_channels * groups)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (out_channels % groups != 0)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int group_in_channels = in_channels;
    int group_out_channels = out_channels / groups;

    int out_h, out_w;
    conv2d_output_shape(H, W, kH, kW,
                        params.pad[0], params.pad[1],
                        params.stride[0], params.stride[1],
                        params.dilation[0], params.dilation[1],
                        &out_h, &out_w);
    if (out_h <= 0 || out_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (output->ndim != 4 ||
        output->dims[0] != N ||
        output->dims[1] != out_channels ||
        output->dims[2] != out_h ||
        output->dims[3] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[4], w_strides[4], out_strides[4];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(output, out_strides);

    util_clear_tensor(output);

    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < groups; g++)
        {
            for (int oc = 0; oc < group_out_channels; oc++)
            {
                int out_c = g * group_out_channels + oc;
                for (int oy = 0; oy < out_h; oy++)
                {
                    for (int ox = 0; ox < out_w; ox++)
                    {
                        float sum = 0.0f;
                        int in_y0 = oy * params.stride[0] - params.pad[0];
                        int in_x0 = ox * params.stride[1] - params.pad[1];
                        for (int ic = 0; ic < group_in_channels; ic++)
                        {
                            int in_c = g * group_in_channels + ic;
                            for (int ky = 0; ky < kH; ky++)
                            {
                                int in_y = in_y0 + ky * params.dilation[0];
                                if (in_y < 0 || in_y >= H)
                                    continue;
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int in_x = in_x0 + kx * params.dilation[1];
                                    if (in_x < 0 || in_x >= W)
                                        continue;
                                    size_t in_off = n * in_strides[0] +
                                                    in_c * in_strides[1] +
                                                    in_y * in_strides[2] +
                                                    in_x * in_strides[3];
                                    size_t w_off = out_c * w_strides[0] +
                                                   ic * w_strides[1] +
                                                   ky * w_strides[2] +
                                                   kx * w_strides[3];
                                    sum += input->data[in_off] * weight->data[w_off];
                                }
                            }
                        }
                        if (bias)
                        {
                            if (bias->ndim == 1 && (int)bias->size == out_channels)
                                sum += bias->data[out_c];
                            else if (bias->ndim == 0)
                                sum += bias->data[0];
                            else
                                return TENSOR_ERR_SHAPE_MISMATCH;
                        }
                        size_t out_off = n * out_strides[0] +
                                         out_c * out_strides[1] +
                                         oy * out_strides[2] +
                                         ox * out_strides[3];
                        output->data[out_off] = sum;
                    }
                }
            }
        }
    }
    return TENSOR_OK;
}

/**
 * @brief 三维卷积
 * @param input  输入张量 [N, C, D, H, W]
 * @param weight 卷积核 [out_channels, in_channels/groups, kD, kH, kW]
 * @param bias   偏置 [out_channels] 或标量，可为 NULL
 * @param params 卷积参数（pad[0..2], stride[0..2], dilation[0..2], groups）
 * @param output 输出张量 [N, out_channels, out_d, out_h, out_w]
 * @return TensorStatus
 */
TensorStatus tensor_conv3d(const Tensor *input, const Tensor *weight,
                           const Tensor *bias, ConvParams params,
                           Tensor *output)
{
    if (!input || !weight || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 5 || weight->ndim != 5)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (params.pad[0] < 0 || params.pad[1] < 0 || params.pad[2] < 0 ||
        params.stride[0] <= 0 || params.stride[1] <= 0 || params.stride[2] <= 0 ||
        params.dilation[0] <= 0 || params.dilation[1] <= 0 || params.dilation[2] <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (params.groups <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    int N = input->dims[0];
    int C = input->dims[1];
    int D = input->dims[2];
    int H = input->dims[3];
    int W = input->dims[4];
    int out_channels = weight->dims[0];
    int in_channels = weight->dims[1];
    int kD = weight->dims[2];
    int kH = weight->dims[3];
    int kW = weight->dims[4];
    int groups = params.groups;

    if (C != in_channels * groups)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (out_channels % groups != 0)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int group_in_channels = in_channels;
    int group_out_channels = out_channels / groups;

    int out_d, out_h, out_w;
    conv3d_output_shape(D, H, W,
                        kD, kH, kW,
                        params.pad[0], params.pad[1], params.pad[2],
                        params.stride[0], params.stride[1], params.stride[2],
                        params.dilation[0], params.dilation[1], params.dilation[2],
                        &out_d, &out_h, &out_w);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (output->ndim != 5 ||
        output->dims[0] != N ||
        output->dims[1] != out_channels ||
        output->dims[2] != out_d ||
        output->dims[3] != out_h ||
        output->dims[4] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[5], w_strides[5], out_strides[5];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(output, out_strides);

    util_clear_tensor(output);

    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < groups; g++)
        {
            for (int oc = 0; oc < group_out_channels; oc++)
            {
                int out_c = g * group_out_channels + oc;
                for (int od = 0; od < out_d; od++)
                {
                    int in_d0 = od * params.stride[0] - params.pad[0];
                    for (int oh = 0; oh < out_h; oh++)
                    {
                        int in_h0 = oh * params.stride[1] - params.pad[1];
                        for (int ow = 0; ow < out_w; ow++)
                        {
                            int in_w0 = ow * params.stride[2] - params.pad[2];
                            float sum = 0.0f;
                            for (int ic = 0; ic < group_in_channels; ic++)
                            {
                                int in_c = g * group_in_channels + ic;
                                for (int kd = 0; kd < kD; kd++)
                                {
                                    int in_d = in_d0 + kd * params.dilation[0];
                                    if (in_d < 0 || in_d >= D)
                                        continue;
                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        int in_h = in_h0 + kh * params.dilation[1];
                                        if (in_h < 0 || in_h >= H)
                                            continue;
                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int in_w = in_w0 + kw * params.dilation[2];
                                            if (in_w < 0 || in_w >= W)
                                                continue;
                                            size_t in_off = n * in_strides[0] +
                                                            in_c * in_strides[1] +
                                                            in_d * in_strides[2] +
                                                            in_h * in_strides[3] +
                                                            in_w * in_strides[4];
                                            size_t w_off = out_c * w_strides[0] +
                                                           ic * w_strides[1] +
                                                           kd * w_strides[2] +
                                                           kh * w_strides[3] +
                                                           kw * w_strides[4];
                                            sum += input->data[in_off] * weight->data[w_off];
                                        }
                                    }
                                }
                            }
                            if (bias)
                            {
                                if (bias->ndim == 1 && (int)bias->size == out_channels)
                                    sum += bias->data[out_c];
                                else if (bias->ndim == 0)
                                    sum += bias->data[0];
                                else
                                    return TENSOR_ERR_SHAPE_MISMATCH;
                            }
                            size_t out_off = n * out_strides[0] +
                                             out_c * out_strides[1] +
                                             od * out_strides[2] +
                                             oh * out_strides[3] +
                                             ow * out_strides[4];
                            output->data[out_off] = sum;
                        }
                    }
                }
            }
        }
    }
    return TENSOR_OK;
}

/* ==================== 转置卷积 ==================== */

/**
 * @brief 一维转置卷积
 * @param input  输入张量 [N, C_in, L]
 * @param weight 卷积核 [out_channels, in_channels/groups, kL] （注意：out_channels 实际是权重的输出通道，应等于输入的通道数）
 * @param bias   偏置 [in_channels] 或标量，可为 NULL
 * @param params 卷积参数
 * @param output 输出张量 [N, in_channels, out_len]
 * @return TensorStatus
 */
TensorStatus tensor_conv_transpose1d(const Tensor *input, const Tensor *weight,
                                     const Tensor *bias, ConvParams params,
                                     Tensor *output)
{
    if (!input || !weight || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 3 || weight->ndim != 3)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (params.pad[0] < 0 || params.stride[0] <= 0 || params.dilation[0] <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (params.groups <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    int N = input->dims[0];
    int C_in = input->dims[1]; // 输入通道数，应等于 weight->dims[0] (输出通道)
    int L = input->dims[2];
    int out_channels = weight->dims[0]; // 权重输出通道，应等于输入通道
    int in_channels = weight->dims[1];  // 权重输入通道，应等于输出通道
    int kL = weight->dims[2];
    int groups = params.groups;

    if (C_in != out_channels * groups) // 输入通道必须等于权重输出通道 * groups
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (in_channels % groups != 0)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int group_in_channels = in_channels / groups; // 每组权重的输入通道
    int group_out_channels = out_channels;        // 每组权重的输出通道 (每组输出通道相同)

    int out_len;
    conv_transpose1d_output_shape(L, kL,
                                  params.pad[0], params.stride[0], params.dilation[0],
                                  &out_len);

    if (output->ndim != 3 ||
        output->dims[0] != N ||
        output->dims[1] != in_channels || // 输出通道应为权重的输入通道
        output->dims[2] != out_len)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[3], w_strides[3], out_strides[3];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(output, out_strides);

    util_clear_tensor(output);

    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < groups; g++)
        {
            for (int ic = 0; ic < group_in_channels; ic++)
            {                                           // 权重的输入通道，也是输出的实际通道
                int out_c = g * group_in_channels + ic; // 输出通道索引
                for (int oc = 0; oc < group_out_channels; oc++)
                {                                           // 权重的输出通道，即输入的通道
                    int in_c = g * group_out_channels + oc; // 输入通道索引
                    for (int il = 0; il < L; il++)
                    {
                        int out_l_base = il * params.stride[0] - params.pad[0];
                        for (int kl = 0; kl < kL; kl++)
                        {
                            int out_l = out_l_base + kl * params.dilation[0];
                            if (out_l < 0 || out_l >= out_len)
                                continue;

                            size_t in_off = n * in_strides[0] +
                                            in_c * in_strides[1] +
                                            il * in_strides[2];
                            size_t w_off = oc * w_strides[0] + // 注意权重索引：oc 是输出通道，ic 是输入通道
                                           ic * w_strides[1] +
                                           kl * w_strides[2];
                            size_t out_off = n * out_strides[0] +
                                             out_c * out_strides[1] +
                                             out_l * out_strides[2];
                            output->data[out_off] += input->data[in_off] * weight->data[w_off];
                        }
                    }
                }
            }
        }
    }

    if (bias)
    {
        if (bias->ndim == 1 && (int)bias->size == in_channels)
        {
            for (int n = 0; n < N; n++)
            {
                for (int c = 0; c < in_channels; c++)
                {
                    for (int ol = 0; ol < out_len; ol++)
                    {
                        size_t out_off = n * out_strides[0] +
                                         c * out_strides[1] +
                                         ol * out_strides[2];
                        output->data[out_off] += bias->data[c];
                    }
                }
            }
        }
        else if (bias->ndim == 0)
        {
            float b = bias->data[0];
            for (size_t i = 0; i < output->size; i++)
                output->data[i] += b;
        }
        else
        {
            return TENSOR_ERR_SHAPE_MISMATCH;
        }
    }

    return TENSOR_OK;
}

/**
 * @brief 二维转置卷积
 */
TensorStatus tensor_conv_transpose2d(const Tensor *input, const Tensor *weight,
                                     const Tensor *bias, ConvParams params,
                                     Tensor *output)
{
    if (!input || !weight || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 4 || weight->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (params.pad[0] < 0 || params.pad[1] < 0 ||
        params.stride[0] <= 0 || params.stride[1] <= 0 ||
        params.dilation[0] <= 0 || params.dilation[1] <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (params.groups <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    int N = input->dims[0];
    int C_in = input->dims[1];
    int H = input->dims[2];
    int W = input->dims[3];
    int out_channels = weight->dims[0];
    int in_channels = weight->dims[1];
    int kH = weight->dims[2];
    int kW = weight->dims[3];
    int groups = params.groups;

    if (C_in != out_channels * groups)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (in_channels % groups != 0)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels;

    int out_h, out_w;
    conv_transpose2d_output_shape(H, W,
                                  kH, kW,
                                  params.pad[0], params.pad[1],
                                  params.stride[0], params.stride[1],
                                  params.dilation[0], params.dilation[1],
                                  &out_h, &out_w);

    if (output->ndim != 4 ||
        output->dims[0] != N ||
        output->dims[1] != in_channels ||
        output->dims[2] != out_h ||
        output->dims[3] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[4], w_strides[4], out_strides[4];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(output, out_strides);

    util_clear_tensor(output);

    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < groups; g++)
        {
            for (int ic = 0; ic < group_in_channels; ic++)
            {
                int out_c = g * group_in_channels + ic;
                for (int oc = 0; oc < group_out_channels; oc++)
                {
                    int in_c = g * group_out_channels + oc;
                    for (int ih = 0; ih < H; ih++)
                    {
                        int out_h_base = ih * params.stride[0] - params.pad[0];
                        for (int iw = 0; iw < W; iw++)
                        {
                            int out_w_base = iw * params.stride[1] - params.pad[1];
                            for (int kh = 0; kh < kH; kh++)
                            {
                                int out_hh = out_h_base + kh * params.dilation[0];
                                if (out_hh < 0 || out_hh >= out_h)
                                    continue;
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int out_ww = out_w_base + kw * params.dilation[1];
                                    if (out_ww < 0 || out_ww >= out_w)
                                        continue;

                                    size_t in_off = n * in_strides[0] +
                                                    in_c * in_strides[1] +
                                                    ih * in_strides[2] +
                                                    iw * in_strides[3];
                                    size_t w_off = oc * w_strides[0] +
                                                   ic * w_strides[1] +
                                                   kh * w_strides[2] +
                                                   kw * w_strides[3];
                                    size_t out_off = n * out_strides[0] +
                                                     out_c * out_strides[1] +
                                                     out_hh * out_strides[2] +
                                                     out_ww * out_strides[3];
                                    output->data[out_off] += input->data[in_off] * weight->data[w_off];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias)
    {
        if (bias->ndim == 1 && (int)bias->size == in_channels)
        {
            for (int n = 0; n < N; n++)
                for (int c = 0; c < in_channels; c++)
                    for (int oh = 0; oh < out_h; oh++)
                        for (int ow = 0; ow < out_w; ow++)
                        {
                            size_t out_off = n * out_strides[0] +
                                             c * out_strides[1] +
                                             oh * out_strides[2] +
                                             ow * out_strides[3];
                            output->data[out_off] += bias->data[c];
                        }
        }
        else if (bias->ndim == 0)
        {
            float b = bias->data[0];
            for (size_t i = 0; i < output->size; i++)
                output->data[i] += b;
        }
        else
        {
            return TENSOR_ERR_SHAPE_MISMATCH;
        }
    }

    return TENSOR_OK;
}

/**
 * @brief 三维转置卷积
 */
TensorStatus tensor_conv_transpose3d(const Tensor *input, const Tensor *weight,
                                     const Tensor *bias, ConvParams params,
                                     Tensor *output)
{
    if (!input || !weight || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 5 || weight->ndim != 5)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (params.pad[0] < 0 || params.pad[1] < 0 || params.pad[2] < 0 ||
        params.stride[0] <= 0 || params.stride[1] <= 0 || params.stride[2] <= 0 ||
        params.dilation[0] <= 0 || params.dilation[1] <= 0 || params.dilation[2] <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (params.groups <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    int N = input->dims[0];
    int C_in = input->dims[1];
    int D = input->dims[2];
    int H = input->dims[3];
    int W = input->dims[4];
    int out_channels = weight->dims[0];
    int in_channels = weight->dims[1];
    int kD = weight->dims[2];
    int kH = weight->dims[3];
    int kW = weight->dims[4];
    int groups = params.groups;

    if (C_in != out_channels * groups)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (in_channels % groups != 0)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels;

    int out_d, out_h, out_w;
    conv_transpose3d_output_shape(D, H, W,
                                  kD, kH, kW,
                                  params.pad[0], params.pad[1], params.pad[2],
                                  params.stride[0], params.stride[1], params.stride[2],
                                  params.dilation[0], params.dilation[1], params.dilation[2],
                                  &out_d, &out_h, &out_w);

    if (output->ndim != 5 ||
        output->dims[0] != N ||
        output->dims[1] != in_channels ||
        output->dims[2] != out_d ||
        output->dims[3] != out_h ||
        output->dims[4] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[5], w_strides[5], out_strides[5];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(output, out_strides);

    util_clear_tensor(output);

    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < groups; g++)
        {
            for (int ic = 0; ic < group_in_channels; ic++)
            {
                int out_c = g * group_in_channels + ic;
                for (int oc = 0; oc < group_out_channels; oc++)
                {
                    int in_c = g * group_out_channels + oc;
                    for (int id = 0; id < D; id++)
                    {
                        int out_d_base = id * params.stride[0] - params.pad[0];
                        for (int ih = 0; ih < H; ih++)
                        {
                            int out_h_base = ih * params.stride[1] - params.pad[1];
                            for (int iw = 0; iw < W; iw++)
                            {
                                int out_w_base = iw * params.stride[2] - params.pad[2];
                                for (int kd = 0; kd < kD; kd++)
                                {
                                    int out_dd = out_d_base + kd * params.dilation[0];
                                    if (out_dd < 0 || out_dd >= out_d)
                                        continue;
                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        int out_hh = out_h_base + kh * params.dilation[1];
                                        if (out_hh < 0 || out_hh >= out_h)
                                            continue;
                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int out_ww = out_w_base + kw * params.dilation[2];
                                            if (out_ww < 0 || out_ww >= out_w)
                                                continue;

                                            size_t in_off = n * in_strides[0] +
                                                            in_c * in_strides[1] +
                                                            id * in_strides[2] +
                                                            ih * in_strides[3] +
                                                            iw * in_strides[4];
                                            size_t w_off = oc * w_strides[0] +
                                                           ic * w_strides[1] +
                                                           kd * w_strides[2] +
                                                           kh * w_strides[3] +
                                                           kw * w_strides[4];
                                            size_t out_off = n * out_strides[0] +
                                                             out_c * out_strides[1] +
                                                             out_dd * out_strides[2] +
                                                             out_hh * out_strides[3] +
                                                             out_ww * out_strides[4];
                                            output->data[out_off] += input->data[in_off] * weight->data[w_off];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias)
    {
        if (bias->ndim == 1 && (int)bias->size == in_channels)
        {
            for (int n = 0; n < N; n++)
                for (int c = 0; c < in_channels; c++)
                    for (int od = 0; od < out_d; od++)
                        for (int oh = 0; oh < out_h; oh++)
                            for (int ow = 0; ow < out_w; ow++)
                            {
                                size_t out_off = n * out_strides[0] +
                                                 c * out_strides[1] +
                                                 od * out_strides[2] +
                                                 oh * out_strides[3] +
                                                 ow * out_strides[4];
                                output->data[out_off] += bias->data[c];
                            }
        }
        else if (bias->ndim == 0)
        {
            float b = bias->data[0];
            for (size_t i = 0; i < output->size; i++)
                output->data[i] += b;
        }
        else
        {
            return TENSOR_ERR_SHAPE_MISMATCH;
        }
    }

    return TENSOR_OK;
}

/* ==================== 池化层（通用实现） ==================== */

/**
 * @brief 一维池化通用实现
 */
static TensorStatus pool1d_general(const Tensor *input, PoolType type,
                                   PoolParams params, Tensor *output)
{
    if (!input || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 3)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int N = input->dims[0];
    int C = input->dims[1];
    int L = input->dims[2];

    int kL = params.kernel[0];
    int padL = params.pad[0];
    int strideL = params.stride[0];
    int ceil_mode = params.ceil_mode;

    int out_len;
    if (ceil_mode)
        out_len = (int)ceil((L + 2 * padL - kL) / (float)strideL) + 1;
    else
        out_len = (L + 2 * padL - kL) / strideL + 1;
    if (out_len <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (output->ndim != 3 ||
        output->dims[0] != N ||
        output->dims[1] != C ||
        output->dims[2] != out_len)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[3], out_strides[3];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(output, out_strides);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int ol = 0; ol < out_len; ol++)
            {
                int in_l_start = ol * strideL - padL;
                int in_l_end = in_l_start + kL;
                float res;
                if (type == POOL_MAX)
                {
                    res = -INFINITY;
                    for (int il = in_l_start; il < in_l_end; il++)
                    {
                        if (il < 0 || il >= L)
                            continue;
                        size_t in_off = n * in_strides[0] +
                                        c * in_strides[1] +
                                        il * in_strides[2];
                        float val = input->data[in_off];
                        if (val > res)
                            res = val;
                    }
                }
                else if (type == POOL_AVG)
                {
                    double sum = 0.0;
                    int count = 0;
                    for (int il = in_l_start; il < in_l_end; il++)
                    {
                        if (il < 0 || il >= L)
                        {
                            if (params.count_include_pad)
                                count++;
                            continue;
                        }
                        size_t in_off = n * in_strides[0] +
                                        c * in_strides[1] +
                                        il * in_strides[2];
                        sum += input->data[in_off];
                        count++;
                    }
                    res = (count == 0) ? 0.0f : (float)(sum / count);
                }
                else
                { // POOL_L2
                    double sum_sq = 0.0;
                    for (int il = in_l_start; il < in_l_end; il++)
                    {
                        if (il < 0 || il >= L)
                            continue;
                        size_t in_off = n * in_strides[0] +
                                        c * in_strides[1] +
                                        il * in_strides[2];
                        float val = input->data[in_off];
                        sum_sq += (double)val * val;
                    }
                    res = (float)sqrt(sum_sq);
                }
                size_t out_off = n * out_strides[0] +
                                 c * out_strides[1] +
                                 ol * out_strides[2];
                output->data[out_off] = res;
            }
        }
    }
    return TENSOR_OK;
}

/**
 * @brief 二维池化通用实现
 */
static TensorStatus pool2d_general(const Tensor *input, PoolType type,
                                   PoolParams params, Tensor *output)
{
    if (!input || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int N = input->dims[0];
    int C = input->dims[1];
    int H = input->dims[2];
    int W = input->dims[3];

    int kH = params.kernel[0];
    int kW = params.kernel[1];
    int padH = params.pad[0];
    int padW = params.pad[1];
    int strideH = params.stride[0];
    int strideW = params.stride[1];
    int ceil_mode = params.ceil_mode;

    int out_h, out_w;
    if (ceil_mode)
    {
        out_h = (int)ceil((H + 2 * padH - kH) / (float)strideH) + 1;
        out_w = (int)ceil((W + 2 * padW - kW) / (float)strideW) + 1;
    }
    else
    {
        out_h = (H + 2 * padH - kH) / strideH + 1;
        out_w = (W + 2 * padW - kW) / strideW + 1;
    }
    if (out_h <= 0 || out_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (output->ndim != 4 ||
        output->dims[0] != N ||
        output->dims[1] != C ||
        output->dims[2] != out_h ||
        output->dims[3] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[4], out_strides[4];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(output, out_strides);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int oh = 0; oh < out_h; oh++)
            {
                for (int ow = 0; ow < out_w; ow++)
                {
                    int in_h_start = oh * strideH - padH;
                    int in_h_end = in_h_start + kH;
                    int in_w_start = ow * strideW - padW;
                    int in_w_end = in_w_start + kW;
                    float res;
                    if (type == POOL_MAX)
                    {
                        res = -INFINITY;
                        for (int ih = in_h_start; ih < in_h_end; ih++)
                        {
                            if (ih < 0 || ih >= H)
                                continue;
                            for (int iw = in_w_start; iw < in_w_end; iw++)
                            {
                                if (iw < 0 || iw >= W)
                                    continue;
                                size_t in_off = n * in_strides[0] +
                                                c * in_strides[1] +
                                                ih * in_strides[2] +
                                                iw * in_strides[3];
                                float val = input->data[in_off];
                                if (val > res)
                                    res = val;
                            }
                        }
                    }
                    else if (type == POOL_AVG)
                    {
                        double sum = 0.0;
                        int count = 0;
                        for (int ih = in_h_start; ih < in_h_end; ih++)
                        {
                            for (int iw = in_w_start; iw < in_w_end; iw++)
                            {
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                {
                                    if (params.count_include_pad)
                                        count++;
                                    continue;
                                }
                                size_t in_off = n * in_strides[0] +
                                                c * in_strides[1] +
                                                ih * in_strides[2] +
                                                iw * in_strides[3];
                                sum += input->data[in_off];
                                count++;
                            }
                        }
                        res = (count == 0) ? 0.0f : (float)(sum / count);
                    }
                    else
                    { // POOL_L2
                        double sum_sq = 0.0;
                        for (int ih = in_h_start; ih < in_h_end; ih++)
                        {
                            for (int iw = in_w_start; iw < in_w_end; iw++)
                            {
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                    continue;
                                size_t in_off = n * in_strides[0] +
                                                c * in_strides[1] +
                                                ih * in_strides[2] +
                                                iw * in_strides[3];
                                float val = input->data[in_off];
                                sum_sq += (double)val * val;
                            }
                        }
                        res = (float)sqrt(sum_sq);
                    }
                    size_t out_off = n * out_strides[0] +
                                     c * out_strides[1] +
                                     oh * out_strides[2] +
                                     ow * out_strides[3];
                    output->data[out_off] = res;
                }
            }
        }
    }
    return TENSOR_OK;
}

/**
 * @brief 三维池化通用实现
 */
static TensorStatus pool3d_general(const Tensor *input, PoolType type,
                                   PoolParams params, Tensor *output)
{
    if (!input || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 5)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int N = input->dims[0];
    int C = input->dims[1];
    int D = input->dims[2];
    int H = input->dims[3];
    int W = input->dims[4];

    int kD = params.kernel[0];
    int kH = params.kernel[1];
    int kW = params.kernel[2];
    int padD = params.pad[0];
    int padH = params.pad[1];
    int padW = params.pad[2];
    int strideD = params.stride[0];
    int strideH = params.stride[1];
    int strideW = params.stride[2];
    int ceil_mode = params.ceil_mode;

    int out_d, out_h, out_w;
    if (ceil_mode)
    {
        out_d = (int)ceil((D + 2 * padD - kD) / (float)strideD) + 1;
        out_h = (int)ceil((H + 2 * padH - kH) / (float)strideH) + 1;
        out_w = (int)ceil((W + 2 * padW - kW) / (float)strideW) + 1;
    }
    else
    {
        out_d = (D + 2 * padD - kD) / strideD + 1;
        out_h = (H + 2 * padH - kH) / strideH + 1;
        out_w = (W + 2 * padW - kW) / strideW + 1;
    }
    if (out_d <= 0 || out_h <= 0 || out_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (output->ndim != 5 ||
        output->dims[0] != N ||
        output->dims[1] != C ||
        output->dims[2] != out_d ||
        output->dims[3] != out_h ||
        output->dims[4] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[5], out_strides[5];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(output, out_strides);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int od = 0; od < out_d; od++)
            {
                int in_d_start = od * strideD - padD;
                int in_d_end = in_d_start + kD;
                for (int oh = 0; oh < out_h; oh++)
                {
                    int in_h_start = oh * strideH - padH;
                    int in_h_end = in_h_start + kH;
                    for (int ow = 0; ow < out_w; ow++)
                    {
                        int in_w_start = ow * strideW - padW;
                        int in_w_end = in_w_start + kW;
                        float res;
                        if (type == POOL_MAX)
                        {
                            res = -INFINITY;
                            for (int id = in_d_start; id < in_d_end; id++)
                            {
                                if (id < 0 || id >= D)
                                    continue;
                                for (int ih = in_h_start; ih < in_h_end; ih++)
                                {
                                    if (ih < 0 || ih >= H)
                                        continue;
                                    for (int iw = in_w_start; iw < in_w_end; iw++)
                                    {
                                        if (iw < 0 || iw >= W)
                                            continue;
                                        size_t in_off = n * in_strides[0] +
                                                        c * in_strides[1] +
                                                        id * in_strides[2] +
                                                        ih * in_strides[3] +
                                                        iw * in_strides[4];
                                        float val = input->data[in_off];
                                        if (val > res)
                                            res = val;
                                    }
                                }
                            }
                        }
                        else if (type == POOL_AVG)
                        {
                            double sum = 0.0;
                            int count = 0;
                            for (int id = in_d_start; id < in_d_end; id++)
                            {
                                for (int ih = in_h_start; ih < in_h_end; ih++)
                                {
                                    for (int iw = in_w_start; iw < in_w_end; iw++)
                                    {
                                        if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= W)
                                        {
                                            if (params.count_include_pad)
                                                count++;
                                            continue;
                                        }
                                        size_t in_off = n * in_strides[0] +
                                                        c * in_strides[1] +
                                                        id * in_strides[2] +
                                                        ih * in_strides[3] +
                                                        iw * in_strides[4];
                                        sum += input->data[in_off];
                                        count++;
                                    }
                                }
                            }
                            res = (count == 0) ? 0.0f : (float)(sum / count);
                        }
                        else
                        { // POOL_L2
                            double sum_sq = 0.0;
                            for (int id = in_d_start; id < in_d_end; id++)
                            {
                                for (int ih = in_h_start; ih < in_h_end; ih++)
                                {
                                    for (int iw = in_w_start; iw < in_w_end; iw++)
                                    {
                                        if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= W)
                                            continue;
                                        size_t in_off = n * in_strides[0] +
                                                        c * in_strides[1] +
                                                        id * in_strides[2] +
                                                        ih * in_strides[3] +
                                                        iw * in_strides[4];
                                        float val = input->data[in_off];
                                        sum_sq += (double)val * val;
                                    }
                                }
                            }
                            res = (float)sqrt(sum_sq);
                        }
                        size_t out_off = n * out_strides[0] +
                                         c * out_strides[1] +
                                         od * out_strides[2] +
                                         oh * out_strides[3] +
                                         ow * out_strides[4];
                        output->data[out_off] = res;
                    }
                }
            }
        }
    }
    return TENSOR_OK;
}

/* ---------- 池化 API ---------- */

TensorStatus tensor_pool1d(const Tensor *input, PoolType type,
                           PoolParams params, Tensor *output)
{
    return pool1d_general(input, type, params, output);
}

TensorStatus tensor_pool2d(const Tensor *input, PoolType type,
                           PoolParams params, Tensor *output)
{
    return pool2d_general(input, type, params, output);
}

TensorStatus tensor_pool3d(const Tensor *input, PoolType type,
                           PoolParams params, Tensor *output)
{
    return pool3d_general(input, type, params, output);
}

/**
 * @brief 二维全局平均池化
 */
TensorStatus tensor_global_avg_pool2d(const Tensor *input, Tensor *output)
{
    if (!input || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int N = input->dims[0];
    int C = input->dims[1];
    int H = input->dims[2];
    int W = input->dims[3];

    if (output->ndim != 4 ||
        output->dims[0] != N ||
        output->dims[1] != C ||
        output->dims[2] != 1 ||
        output->dims[3] != 1)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[4], out_strides[4];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(output, out_strides);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            double sum = 0.0;
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    size_t in_off = n * in_strides[0] +
                                    c * in_strides[1] +
                                    h * in_strides[2] +
                                    w * in_strides[3];
                    sum += input->data[in_off];
                }
            }
            size_t out_off = n * out_strides[0] +
                             c * out_strides[1];
            output->data[out_off] = (float)(sum / (H * W));
        }
    }
    return TENSOR_OK;
}

/**
 * @brief 二维全局最大池化
 */
TensorStatus tensor_global_max_pool2d(const Tensor *input, Tensor *output)
{
    if (!input || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int N = input->dims[0];
    int C = input->dims[1];
    int H = input->dims[2];
    int W = input->dims[3];

    if (output->ndim != 4 ||
        output->dims[0] != N ||
        output->dims[1] != C ||
        output->dims[2] != 1 ||
        output->dims[3] != 1)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[4], out_strides[4];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(output, out_strides);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            float max_val = -INFINITY;
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    size_t in_off = n * in_strides[0] +
                                    c * in_strides[1] +
                                    h * in_strides[2] +
                                    w * in_strides[3];
                    float val = input->data[in_off];
                    if (val > max_val)
                        max_val = val;
                }
            }
            size_t out_off = n * out_strides[0] +
                             c * out_strides[1];
            output->data[out_off] = max_val;
        }
    }
    return TENSOR_OK;
}

/* ==================== 归一化层 ==================== */

/**
 * @brief 批归一化（推理模式）
 */
TensorStatus tensor_batchnorm(const Tensor *x, const Tensor *mean,
                              const Tensor *var, const Tensor *scale,
                              const Tensor *bias, float epsilon,
                              Tensor *y)
{
    if (!x || !mean || !var || !y)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim < 2)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int C = x->dims[1];
    if ((mean->ndim != 1 || (int)mean->size != C) &&
        !(mean->ndim == 0 && C == 1))
        return TENSOR_ERR_SHAPE_MISMATCH;
    if ((var->ndim != 1 || (int)var->size != C) &&
        !(var->ndim == 0 && C == 1))
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (scale && ((scale->ndim != 1 || (int)scale->size != C) &&
                  !(scale->ndim == 0 && C == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (bias && ((bias->ndim != 1 || (int)bias->size != C) &&
                 !(bias->ndim == 0 && C == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (y->ndim != x->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != y->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(y);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], y_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(y, y_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_off = 0, y_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            x_off += coords[i] * x_strides[i];
            y_off += coords[i] * y_strides[i];
        }
        int c = coords[1];
        float m = (mean->ndim == 0) ? mean->data[0] : mean->data[c];
        float v = (var->ndim == 0) ? var->data[0] : var->data[c];
        float s = scale ? ((scale->ndim == 0) ? scale->data[0] : scale->data[c]) : 1.0f;
        float b = bias ? ((bias->ndim == 0) ? bias->data[0] : bias->data[c]) : 0.0f;

        float x_val = x->data[x_off];
        float x_norm = (x_val - m) / sqrtf(v + epsilon);
        y->data[y_off] = s * x_norm + b;

        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/**
 * @brief 层归一化
 */
TensorStatus tensor_layernorm(const Tensor *x, const Tensor *scale,
                              const Tensor *bias, float epsilon,
                              Tensor *y)
{
    if (!x || !y)
        return TENSOR_ERR_NULL_PTR;
    int last_dim = x->dims[x->ndim - 1];
    if (scale && ((scale->ndim != 1 || (int)scale->size != last_dim) &&
                  !(scale->ndim == 0 && last_dim == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (bias && ((bias->ndim != 1 || (int)bias->size != last_dim) &&
                 !(bias->ndim == 0 && last_dim == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (y->ndim != x->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != y->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(y);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], y_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(y, y_strides);

    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_strides_x[TENSOR_MAX_DIM];
    int outer_strides_y[TENSOR_MAX_DIM];
    for (int i = 0; i < outer_ndim; i++)
    {
        outer_dims[i] = x->dims[i];
        outer_strides_x[i] = x_strides[i];
        outer_strides_y[i] = y_strides[i];
    }

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_base = 0, y_base = 0;
        for (int i = 0; i < outer_ndim; i++)
        {
            x_base += outer_coords[i] * outer_strides_x[i];
            y_base += outer_coords[i] * outer_strides_y[i];
        }

        double sum = 0.0;
        for (int j = 0; j < last_dim; j++)
        {
            size_t x_off = x_base + j * x_strides[outer_ndim];
            sum += x->data[x_off];
        }
        double mean = sum / last_dim;

        double sq_sum = 0.0;
        for (int j = 0; j < last_dim; j++)
        {
            size_t x_off = x_base + j * x_strides[outer_ndim];
            double d = x->data[x_off] - mean;
            sq_sum += d * d;
        }
        double var = sq_sum / last_dim;
        float inv_std = 1.0f / sqrtf((float)var + epsilon);

        for (int j = 0; j < last_dim; j++)
        {
            size_t x_off = x_base + j * x_strides[outer_ndim];
            size_t y_off = y_base + j * y_strides[outer_ndim];
            float x_val = x->data[x_off];
            float x_norm = (x_val - (float)mean) * inv_std;
            float s = scale ? ((scale->ndim == 0) ? scale->data[0] : scale->data[j]) : 1.0f;
            float b = bias ? ((bias->ndim == 0) ? bias->data[0] : bias->data[j]) : 0.0f;
            y->data[y_off] = s * x_norm + b;
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }
    return TENSOR_OK;
}

/**
 * @brief 实例归一化
 */
TensorStatus tensor_instancenorm(const Tensor *x, const Tensor *scale,
                                 const Tensor *bias, float epsilon,
                                 Tensor *y)
{
    if (!x || !y)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim < 3)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int N = x->dims[0];
    int C = x->dims[1];
    int spatial_ndim = x->ndim - 2;
    int spatial_size = 1;
    for (int i = 2; i < x->ndim; i++)
        spatial_size *= x->dims[i];

    if (scale && ((scale->ndim != 1 || (int)scale->size != C) &&
                  !(scale->ndim == 0 && C == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (bias && ((bias->ndim != 1 || (int)bias->size != C) &&
                 !(bias->ndim == 0 && C == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (y->ndim != x->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != y->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(y);
    if (status != TENSOR_OK)
        return status;

    int x_strides[TENSOR_MAX_DIM], y_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(y, y_strides);

    int spatial_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < spatial_ndim; i++)
        spatial_dims[i] = x->dims[2 + i];

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            double sum = 0.0;
            int coords[TENSOR_MAX_DIM] = {0};
            while (1)
            {
                size_t x_off = n * x_strides[0] + c * x_strides[1];
                for (int d = 0; d < spatial_ndim; d++)
                    x_off += coords[d] * x_strides[2 + d];
                sum += x->data[x_off];

                if (util_increment_coords(coords, spatial_dims, spatial_ndim))
                    break;
            }
            double mean = sum / spatial_size;

            double sq_sum = 0.0;
            memset(coords, 0, spatial_ndim * sizeof(int));
            while (1)
            {
                size_t x_off = n * x_strides[0] + c * x_strides[1];
                for (int d = 0; d < spatial_ndim; d++)
                    x_off += coords[d] * x_strides[2 + d];
                double d = x->data[x_off] - mean;
                sq_sum += d * d;

                if (util_increment_coords(coords, spatial_dims, spatial_ndim))
                    break;
            }
            double var = sq_sum / spatial_size;
            float inv_std = 1.0f / sqrtf((float)var + epsilon);

            float s = scale ? ((scale->ndim == 0) ? scale->data[0] : scale->data[c]) : 1.0f;
            float b = bias ? ((bias->ndim == 0) ? bias->data[0] : bias->data[c]) : 0.0f;

            memset(coords, 0, spatial_ndim * sizeof(int));
            while (1)
            {
                size_t x_off = n * x_strides[0] + c * x_strides[1];
                size_t y_off = n * y_strides[0] + c * y_strides[1];
                for (int d = 0; d < spatial_ndim; d++)
                {
                    x_off += coords[d] * x_strides[2 + d];
                    y_off += coords[d] * y_strides[2 + d];
                }
                float x_val = x->data[x_off];
                float x_norm = (x_val - (float)mean) * inv_std;
                y->data[y_off] = s * x_norm + b;

                if (util_increment_coords(coords, spatial_dims, spatial_ndim))
                    break;
            }
        }
    }
    return TENSOR_OK;
}

/**
 * @brief 组归一化
 */
TensorStatus tensor_groupnorm(const Tensor *x, const Tensor *scale,
                              const Tensor *bias, int num_groups,
                              float epsilon, Tensor *y)
{
    if (!x || !y)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim < 2)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int N = x->dims[0];
    int C = x->dims[1];
    if (C % num_groups != 0)
        return TENSOR_ERR_INVALID_PARAM;
    int group_size = C / num_groups;
    int spatial_ndim = x->ndim - 2;
    int spatial_size = 1;
    for (int i = 2; i < x->ndim; i++)
        spatial_size *= x->dims[i];
    int group_elements = group_size * spatial_size;

    if (scale && ((scale->ndim != 1 || (int)scale->size != C) &&
                  !(scale->ndim == 0 && C == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (bias && ((bias->ndim != 1 || (int)bias->size != C) &&
                 !(bias->ndim == 0 && C == 1)))
        return TENSOR_ERR_SHAPE_MISMATCH;

    if (y->ndim != x->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != y->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(y);
    if (status != TENSOR_OK)
        return status;

    int x_strides[TENSOR_MAX_DIM], y_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(y, y_strides);

    int spatial_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < spatial_ndim; i++)
        spatial_dims[i] = x->dims[2 + i];

    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < num_groups; g++)
        {
            double sum = 0.0;
            for (int gc = 0; gc < group_size; gc++)
            {
                int c = g * group_size + gc;
                int coords[TENSOR_MAX_DIM] = {0};
                while (1)
                {
                    size_t x_off = n * x_strides[0] + c * x_strides[1];
                    for (int d = 0; d < spatial_ndim; d++)
                        x_off += coords[d] * x_strides[2 + d];
                    sum += x->data[x_off];

                    if (util_increment_coords(coords, spatial_dims, spatial_ndim))
                        break;
                }
            }
            double mean = sum / group_elements;

            double sq_sum = 0.0;
            for (int gc = 0; gc < group_size; gc++)
            {
                int c = g * group_size + gc;
                int coords[TENSOR_MAX_DIM] = {0};
                while (1)
                {
                    size_t x_off = n * x_strides[0] + c * x_strides[1];
                    for (int d = 0; d < spatial_ndim; d++)
                        x_off += coords[d] * x_strides[2 + d];
                    double d = x->data[x_off] - mean;
                    sq_sum += d * d;
                    if (util_increment_coords(coords, spatial_dims, spatial_ndim))
                        break;
                }
            }
            double var = sq_sum / group_elements;
            float inv_std = 1.0f / sqrtf((float)var + epsilon);

            for (int gc = 0; gc < group_size; gc++)
            {
                int c = g * group_size + gc;
                float s = scale ? ((scale->ndim == 0) ? scale->data[0] : scale->data[c]) : 1.0f;
                float b = bias ? ((bias->ndim == 0) ? bias->data[0] : bias->data[c]) : 0.0f;
                int coords[TENSOR_MAX_DIM] = {0};
                while (1)
                {
                    size_t x_off = n * x_strides[0] + c * x_strides[1];
                    size_t y_off = n * y_strides[0] + c * y_strides[1];
                    for (int d = 0; d < spatial_ndim; d++)
                    {
                        x_off += coords[d] * x_strides[2 + d];
                        y_off += coords[d] * y_strides[2 + d];
                    }
                    float x_val = x->data[x_off];
                    float x_norm = (x_val - (float)mean) * inv_std;
                    y->data[y_off] = s * x_norm + b;
                    if (util_increment_coords(coords, spatial_dims, spatial_ndim))
                        break;
                }
            }
        }
    }
    return TENSOR_OK;
}

/**
 * @brief 本地响应归一化 (LRN)
 */
TensorStatus tensor_lrn(const Tensor *x, int size, float alpha,
                        float beta, float bias, Tensor *y)
{
    if (!x || !y)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (size % 2 == 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (y->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < 4; i++)
        if (x->dims[i] != y->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(y);
    if (status != TENSOR_OK)
        return status;

    int N = x->dims[0];
    int C = x->dims[1];
    int H = x->dims[2];
    int W = x->dims[3];
    int half = size / 2;

    int x_strides[4], y_strides[4];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(y, y_strides);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    double sq_sum = 0.0;
                    int start_c = c - half;
                    int end_c = c + half;
                    for (int cc = start_c; cc <= end_c; cc++)
                    {
                        if (cc < 0 || cc >= C)
                            continue;
                        size_t in_off = n * x_strides[0] +
                                        cc * x_strides[1] +
                                        h * x_strides[2] +
                                        w * x_strides[3];
                        float val = x->data[in_off];
                        sq_sum += (double)val * val;
                    }
                    float scale = bias + alpha * (float)sq_sum / size;
                    float denom = powf(scale, -beta);

                    size_t in_off = n * x_strides[0] +
                                    c * x_strides[1] +
                                    h * x_strides[2] +
                                    w * x_strides[3];
                    size_t out_off = n * y_strides[0] +
                                     c * y_strides[1] +
                                     h * y_strides[2] +
                                     w * y_strides[3];
                    y->data[out_off] = x->data[in_off] * denom;
                }
            }
        }
    }
    return TENSOR_OK;
}

/* ==================== 激活函数 ==================== */

/* 基础激活函数（无参） */
static float relu_op(float x) { return x > 0 ? x : 0; }
static float leaky_relu_op(float x, float alpha) { return x > 0 ? x : alpha * x; }
static float elu_op(float x, float alpha) { return x > 0 ? x : alpha * (expf(x) - 1); }
static float selu_op(float x, float alpha, float scale) { return scale * (x > 0 ? x : alpha * (expf(x) - 1)); }
// static float gelu_op(float x)
// {
//     float c = 0.7978845608028654f; // sqrt(2/pi)
//     return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
// }
static float gelu_op(float x)
{
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}
static float swish_op(float x) { return x / (1.0f + expf(-x)); }
// static float mish_op(float x)
// {
//     float sp = logf(1.0f + expf(x));
//     return x * tanhf(sp);
// }
static float mish_op(float x)
{
    double xd = x;
    double sp = log1p(exp(xd)); // log1p 比 log(1+exp) 更精确
    return (float)(xd * tanh(sp));
}
static float softplus_op(float x) { return logf(1.0f + expf(x)); }
static float softsign_op(float x) { return x / (1.0f + fabsf(x)); }
static float hardswish_op(float x)
{
    if (x <= -3)
        return 0;
    if (x >= 3)
        return x;
    return x * (x + 3) / 6.0f;
}
static float hardsigmoid_op(float x)
{
    if (x <= -3)
        return 0;
    if (x >= 3)
        return 1;
    return (x + 3) / 6.0f;
}
static float prelu_op(float x, float alpha) { return x > 0 ? x : alpha * x; }

/**
 * @brief 一元激活函数通用实现（无参）
 */
static TensorStatus unary_activation(const Tensor *x, Tensor *out, float (*op)(float))
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_off = 0, out_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            x_off += coords[i] * x_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        out->data[out_off] = op(x->data[x_off]);

        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_relu(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, relu_op);
}

TensorStatus tensor_gelu(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, gelu_op);
}

TensorStatus tensor_swish(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, swish_op);
}

TensorStatus tensor_mish(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, mish_op);
}

TensorStatus tensor_softplus(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, softplus_op);
}

TensorStatus tensor_softsign(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, softsign_op);
}

TensorStatus tensor_hardswish(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, hardswish_op);
}

TensorStatus tensor_hardsigmoid(const Tensor *x, Tensor *out)
{
    return unary_activation(x, out, hardsigmoid_op);
}

/* 带参数的激活函数 */
TensorStatus tensor_leaky_relu(const Tensor *x, float alpha, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_off = 0, out_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            x_off += coords[i] * x_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        float val = x->data[x_off];
        out->data[out_off] = leaky_relu_op(val, alpha);
        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_elu(const Tensor *x, float alpha, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_off = 0, out_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            x_off += coords[i] * x_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        float val = x->data[x_off];
        out->data[out_off] = elu_op(val, alpha);
        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_selu(const Tensor *x, float alpha, float scale, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_off = 0, out_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            x_off += coords[i] * x_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        float val = x->data[x_off];
        out->data[out_off] = selu_op(val, alpha, scale);
        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

TensorStatus tensor_prelu(const Tensor *x, const Tensor *alpha, Tensor *out)
{
    if (!x || !alpha || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;
    int C = x->dims[1]; // 假设通道在第二维
    if (alpha->ndim != 1 || (int)alpha->size != C)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);
    int alpha_stride = alpha->strides ? alpha->strides[0] : 1;

    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_off = 0, out_off = 0;
        for (int i = 0; i < ndim; i++)
        {
            x_off += coords[i] * x_strides[i];
            out_off += coords[i] * out_strides[i];
        }
        int c = coords[1];
        float a = alpha->data[c * alpha_stride];
        float val = x->data[x_off];
        out->data[out_off] = prelu_op(val, a);
        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/* ==================== 其他层 ==================== */

/**
 * @brief 全连接层
 */
TensorStatus tensor_linear(const Tensor *input, const Tensor *weight,
                           const Tensor *bias, Tensor *output)
{
    if (!input || !weight || !output)
        return TENSOR_ERR_NULL_PTR;
    if (input->ndim < 2)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int in_features = input->dims[input->ndim - 1];
    int out_features = weight->dims[0];
    if (weight->ndim != 2 || weight->dims[1] != in_features)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (bias && (bias->ndim != 1 || (int)bias->size != out_features))
        return TENSOR_ERR_SHAPE_MISMATCH;

    int out_ndim = input->ndim;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < out_ndim - 1; i++)
        out_dims[i] = input->dims[i];
    out_dims[out_ndim - 1] = out_features;

    if (output->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; i++)
        if (output->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(output);
    if (status != TENSOR_OK)
        return status;

    int in_strides[TENSOR_MAX_DIM], w_strides[2], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(output, out_strides);

    int outer_ndim = out_ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_strides_in[TENSOR_MAX_DIM];
    int outer_strides_out[TENSOR_MAX_DIM];
    for (int i = 0; i < outer_ndim; i++)
    {
        outer_dims[i] = input->dims[i];
        outer_strides_in[i] = in_strides[i];
        outer_strides_out[i] = out_strides[i];
    }

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t in_base = 0, out_base = 0;
        for (int i = 0; i < outer_ndim; i++)
        {
            in_base += outer_coords[i] * outer_strides_in[i];
            out_base += outer_coords[i] * outer_strides_out[i];
        }

        for (int oc = 0; oc < out_features; oc++)
        {
            double sum = 0.0;
            for (int ic = 0; ic < in_features; ic++)
            {
                size_t in_off = in_base + ic * in_strides[outer_ndim];
                size_t w_off = oc * w_strides[0] + ic * w_strides[1];
                sum += (double)input->data[in_off] * weight->data[w_off];
            }
            if (bias)
                sum += bias->data[oc];
            size_t out_off = out_base + oc * out_strides[outer_ndim];
            output->data[out_off] = (float)sum;
        }

        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }
    return TENSOR_OK;
}

/**
 * @brief Dropout 层
 */
TensorStatus tensor_dropout(const Tensor *x, float p, int training, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (p < 0 || p >= 1)
        return TENSOR_ERR_INVALID_PARAM;
    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    if (!training)
    {
        return tensor_copy(out, x);
    }

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    float scale = 1.0f / (1.0f - p);
    int coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        ptrdiff_t x_off = util_offset_from_coords(coords, x_strides, ndim);
        ptrdiff_t out_off = util_offset_from_coords(coords, out_strides, ndim);
        float r = (float)rand() / RAND_MAX;
        if (r < p)
            out->data[out_off] = 0.0f;
        else
            out->data[out_off] = x->data[x_off] * scale;

        if (util_increment_coords(coords, x->dims, ndim))
            break;
    }
    return TENSOR_OK;
}

/**
 * @brief Softmax 函数
 */
TensorStatus tensor_softmax(const Tensor *x, int axis, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    axis = util_normalize_axis(axis, x->ndim);
    if (axis < 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_strides_x[TENSOR_MAX_DIM];
    int outer_strides_y[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; i++)
    {
        if (i == axis)
            continue;
        outer_dims[idx] = x->dims[i];
        outer_strides_x[idx] = x_strides[i];
        outer_strides_y[idx] = out_strides[i];
        idx++;
    }

    int inner_size = x->dims[axis];
    int inner_stride_x = x_strides[axis];
    int inner_stride_y = out_strides[axis];

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_base = 0, y_base = 0;
        for (int i = 0; i < outer_ndim; i++)
        {
            x_base += outer_coords[i] * outer_strides_x[i];
            y_base += outer_coords[i] * outer_strides_y[i];
        }

        float max_val = -INFINITY;
        for (int j = 0; j < inner_size; j++)
        {
            size_t x_off = x_base + j * inner_stride_x;
            float val = x->data[x_off];
            if (val > max_val)
                max_val = val;
        }

        if (max_val == -INFINITY)
        {
            float uniform = 1.0f / inner_size;
            for (int j = 0; j < inner_size; j++)
            {
                size_t y_off = y_base + j * inner_stride_y;
                out->data[y_off] = uniform;
            }
        }
        else
        {
            double sum = 0.0;
            for (int j = 0; j < inner_size; j++)
            {
                size_t x_off = x_base + j * inner_stride_x;
                sum += expf(x->data[x_off] - max_val);
            }
            float inv_sum = 1.0f / (float)sum;
            for (int j = 0; j < inner_size; j++)
            {
                size_t x_off = x_base + j * inner_stride_x;
                size_t y_off = y_base + j * inner_stride_y;
                out->data[y_off] = expf(x->data[x_off] - max_val) * inv_sum;
            }
        }
        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }
    return TENSOR_OK;
}

/**
 * @brief LogSoftmax 函数
 */
TensorStatus tensor_log_softmax(const Tensor *x, int axis, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    axis = util_normalize_axis(axis, x->ndim);
    if (axis < 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (x->ndim != out->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < x->ndim; i++)
        if (x->dims[i] != out->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int ndim = x->ndim;
    int x_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int outer_ndim = ndim - 1;
    int outer_dims[TENSOR_MAX_DIM];
    int outer_strides_x[TENSOR_MAX_DIM];
    int outer_strides_y[TENSOR_MAX_DIM];
    int idx = 0;
    for (int i = 0; i < ndim; i++)
    {
        if (i == axis)
            continue;
        outer_dims[idx] = x->dims[i];
        outer_strides_x[idx] = x_strides[i];
        outer_strides_y[idx] = out_strides[i];
        idx++;
    }

    int inner_size = x->dims[axis];
    int inner_stride_x = x_strides[axis];
    int inner_stride_y = out_strides[axis];

    int outer_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        size_t x_base = 0, y_base = 0;
        for (int i = 0; i < outer_ndim; i++)
        {
            x_base += outer_coords[i] * outer_strides_x[i];
            y_base += outer_coords[i] * outer_strides_y[i];
        }

        float max_val = -INFINITY;
        for (int j = 0; j < inner_size; j++)
        {
            size_t x_off = x_base + j * inner_stride_x;
            float val = x->data[x_off];
            if (val > max_val)
                max_val = val;
        }

        if (max_val == -INFINITY)
        {
            float log_uniform = -logf((float)inner_size);
            for (int j = 0; j < inner_size; j++)
            {
                size_t y_off = y_base + j * inner_stride_y;
                out->data[y_off] = log_uniform;
            }
        }
        else
        {
            double sum = 0.0;
            for (int j = 0; j < inner_size; j++)
            {
                size_t x_off = x_base + j * inner_stride_x;
                sum += expf(x->data[x_off] - max_val);
            }
            float log_sum = (float)log(sum);
            for (int j = 0; j < inner_size; j++)
            {
                size_t x_off = x_base + j * inner_stride_x;
                size_t y_off = y_base + j * inner_stride_y;
                out->data[y_off] = (x->data[x_off] - max_val) - log_sum;
            }
        }
        if (util_increment_coords(outer_coords, outer_dims, outer_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ==================== 上采样 ==================== */

/**
 * @brief 2D 上采样
 */
TensorStatus tensor_upsample2d(const Tensor *x, int scale_h, int scale_w,
                               InterpMode mode, int align_corners, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    int N = x->dims[0];
    int C = x->dims[1];
    int H = x->dims[2];
    int W = x->dims[3];
    int out_h = H * scale_h;
    int out_w = W * scale_w;

    if (out->ndim != 4 ||
        out->dims[0] != N ||
        out->dims[1] != C ||
        out->dims[2] != out_h ||
        out->dims[3] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[4], out_strides[4];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    if (mode == UPSAMPLE_NEAREST)
    {
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int oh = 0; oh < out_h; oh++)
                {
                    int ih = oh / scale_h;
                    for (int ow = 0; ow < out_w; ow++)
                    {
                        int iw = ow / scale_w;
                        size_t x_off = n * x_strides[0] +
                                       c * x_strides[1] +
                                       ih * x_strides[2] +
                                       iw * x_strides[3];
                        size_t out_off = n * out_strides[0] +
                                         c * out_strides[1] +
                                         oh * out_strides[2] +
                                         ow * out_strides[3];
                        out->data[out_off] = x->data[x_off];
                    }
                }
            }
        }
        return TENSOR_OK;
    }
    else if (mode == UPSAMPLE_LINEAR)
    {
        float inv_scale_h = 1.0f / scale_h;
        float inv_scale_w = 1.0f / scale_w;
        float h_factor = align_corners ? (H - 1.0f) / (out_h - 1.0f) : inv_scale_h;
        float w_factor = align_corners ? (W - 1.0f) / (out_w - 1.0f) : inv_scale_w;

        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int oh = 0; oh < out_h; oh++)
                {
                    float yf;
                    if (align_corners)
                    {
                        if (oh == 0)
                            yf = 0.0f;
                        else if (oh == out_h - 1)
                            yf = H - 1.0f;
                        else
                            yf = oh * h_factor;
                    }
                    else
                    {
                        yf = (oh + 0.5f) * inv_scale_h - 0.5f;
                    }

                    int y0 = (int)floorf(yf);
                    int y1 = y0 + 1;
                    float dy = yf - y0;

                    if (y0 < 0)
                    {
                        y0 = 0;
                        y1 = 0;
                        dy = 0.0f;
                    }
                    else if (y1 >= H)
                    {
                        y0 = H - 1;
                        y1 = H - 1;
                        dy = 0.0f;
                    }

                    for (int ow = 0; ow < out_w; ow++)
                    {
                        float xf;
                        if (align_corners)
                        {
                            if (ow == 0)
                                xf = 0.0f;
                            else if (ow == out_w - 1)
                                xf = W - 1.0f;
                            else
                                xf = ow * w_factor;
                        }
                        else
                        {
                            xf = (ow + 0.5f) * inv_scale_w - 0.5f;
                        }

                        int x0 = (int)floorf(xf);
                        int x1 = x0 + 1;
                        float dx = xf - x0;

                        if (x0 < 0)
                        {
                            x0 = 0;
                            x1 = 0;
                            dx = 0.0f;
                        }
                        else if (x1 >= W)
                        {
                            x0 = W - 1;
                            x1 = W - 1;
                            dx = 0.0f;
                        }

                        size_t off00 = n * x_strides[0] + c * x_strides[1] + y0 * x_strides[2] + x0 * x_strides[3];
                        size_t off01 = n * x_strides[0] + c * x_strides[1] + y0 * x_strides[2] + x1 * x_strides[3];
                        size_t off10 = n * x_strides[0] + c * x_strides[1] + y1 * x_strides[2] + x0 * x_strides[3];
                        size_t off11 = n * x_strides[0] + c * x_strides[1] + y1 * x_strides[2] + x1 * x_strides[3];

                        float v00 = x->data[off00];
                        float v01 = x->data[off01];
                        float v10 = x->data[off10];
                        float v11 = x->data[off11];

                        float v0 = v00 * (1 - dx) + v01 * dx;
                        float v1 = v10 * (1 - dx) + v11 * dx;
                        float val = v0 * (1 - dy) + v1 * dy;

                        size_t out_off = n * out_strides[0] +
                                         c * out_strides[1] +
                                         oh * out_strides[2] +
                                         ow * out_strides[3];
                        out->data[out_off] = val;
                    }
                }
            }
        }
        return TENSOR_OK;
    }
    else if (mode == UPSAMPLE_CUBIC)
    {
        float inv_scale_h = 1.0f / scale_h;
        float inv_scale_w = 1.0f / scale_w;
        float h_factor = align_corners ? (H - 1.0f) / (out_h - 1.0f) : inv_scale_h;
        float w_factor = align_corners ? (W - 1.0f) / (out_w - 1.0f) : inv_scale_w;

        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int oh = 0; oh < out_h; oh++)
                {
                    float yf = align_corners ? oh * h_factor : (oh + 0.5f) * inv_scale_h - 0.5f;
                    int y0 = (int)floorf(yf) - 1;
                    int y1 = y0 + 1;
                    int y2 = y0 + 2;
                    int y3 = y0 + 3;
                    float dy = yf - floorf(yf);

                    y0 = reflect_coord(y0, H);
                    y1 = reflect_coord(y1, H);
                    y2 = reflect_coord(y2, H);
                    y3 = reflect_coord(y3, H);

                    for (int ow = 0; ow < out_w; ow++)
                    {
                        float xf = align_corners ? ow * w_factor : (ow + 0.5f) * inv_scale_w - 0.5f;
                        int x0 = (int)floorf(xf) - 1;
                        int x1 = x0 + 1;
                        int x2 = x0 + 2;
                        int x3 = x0 + 3;
                        float dx = xf - floorf(xf);

                        x0 = reflect_coord(x0, W);
                        x1 = reflect_coord(x1, W);
                        x2 = reflect_coord(x2, W);
                        x3 = reflect_coord(x3, W);

                        float p[4][4];
                        int y_idx[4] = {y0, y1, y2, y3};
                        int x_idx[4] = {x0, x1, x2, x3};
                        for (int i = 0; i < 4; i++)
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                size_t off = n * x_strides[0] + c * x_strides[1] +
                                             y_idx[i] * x_strides[2] + x_idx[j] * x_strides[3];
                                p[i][j] = x->data[off];
                            }
                        }

                        float row_val[4];
                        for (int i = 0; i < 4; i++)
                            row_val[i] = catmull_rom(p[i][0], p[i][1], p[i][2], p[i][3], dx);

                        float val = catmull_rom(row_val[0], row_val[1], row_val[2], row_val[3], dy);

                        size_t out_off = n * out_strides[0] + c * out_strides[1] +
                                         oh * out_strides[2] + ow * out_strides[3];
                        out->data[out_off] = val;
                    }
                }
            }
        }
        return TENSOR_OK;
    }
    return TENSOR_ERR_INVALID_PARAM; // 未知模式
}

/* ==================== 最大反池化 ==================== */

/**
 * @brief 2D 最大反池化
 */
TensorStatus tensor_max_unpool2d(const Tensor *x, const Tensor *indices,
                                 const int *output_size, Tensor *out)
{
    if (!x || !indices || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 4 || indices->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (output_size == NULL)
        return TENSOR_ERR_INVALID_PARAM;

    int N = x->dims[0];
    int C = x->dims[1];
    int H_in = x->dims[2];
    int W_in = x->dims[3];

    for (int i = 0; i < 4; ++i)
        if (indices->dims[i] != x->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    int out_h = output_size[0];
    int out_w = output_size[1];
    if (out_h <= 0 || out_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 4 || out->dims[0] != N || out->dims[1] != C ||
        out->dims[2] != out_h || out->dims[3] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[4], idx_strides[4], out_strides[4];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(indices, idx_strides);
    util_get_effective_strides(out, out_strides);

    util_clear_tensor(out);

    int coords[4] = {0};
    while (1)
    {
        size_t x_off = 0, idx_off = 0;
        for (int i = 0; i < 4; ++i)
        {
            x_off += coords[i] * x_strides[i];
            idx_off += coords[i] * idx_strides[i];
        }
        float val = x->data[x_off];
        float idx_float = indices->data[idx_off];
        TensorStatus st;
        int idx_val = tensor_float_to_index(idx_float, out_h * out_w, &st);
        if (st != TENSOR_OK)
            return st;
        int out_h_idx = idx_val / out_w;
        int out_w_idx = idx_val % out_w;
        if (out_h_idx < 0 || out_h_idx >= out_h || out_w_idx < 0 || out_w_idx >= out_w)
            return TENSOR_ERR_INDEX_OUT_OF_BOUNDS;

        int out_coords[4] = {coords[0], coords[1], out_h_idx, out_w_idx};
        ptrdiff_t out_off = util_offset_from_coords(out_coords, out_strides, 4);
        out->data[out_off] += val;

        if (util_increment_coords(coords, x->dims, 4))
            break;
    }

    return TENSOR_OK;
}

/* ==================== 自适应池化 ==================== */

/**
 * @brief 2D 自适应平均池化
 */
TensorStatus tensor_adaptive_avg_pool2d(const Tensor *x, const int *output_size, Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 4)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (output_size == NULL)
        return TENSOR_ERR_INVALID_PARAM;

    int N = x->dims[0];
    int C = x->dims[1];
    int H = x->dims[2];
    int W = x->dims[3];
    int out_h = output_size[0];
    int out_w = output_size[1];

    if (out_h <= 0 || out_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;
    if (out_h > H || out_w > W)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 4 || out->dims[0] != N || out->dims[1] != C ||
        out->dims[2] != out_h || out->dims[3] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[4], out_strides[4];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int *h_start = (int *)malloc(out_h * sizeof(int));
    int *h_end = (int *)malloc(out_h * sizeof(int));
    int *w_start = (int *)malloc(out_w * sizeof(int));
    int *w_end = (int *)malloc(out_w * sizeof(int));
    if (!h_start || !h_end || !w_start || !w_end)
    {
        free(h_start);
        free(h_end);
        free(w_start);
        free(w_end);
        return TENSOR_ERR_MEMORY;
    }

    for (int oh = 0; oh < out_h; ++oh)
    {
        h_start[oh] = (int)floorf(oh * H / (float)out_h);
        h_end[oh] = (int)floorf((oh + 1) * H / (float)out_h);
        if (h_end[oh] <= h_start[oh])
            h_end[oh] = h_start[oh] + 1;
    }
    for (int ow = 0; ow < out_w; ++ow)
    {
        w_start[ow] = (int)floorf(ow * W / (float)out_w);
        w_end[ow] = (int)floorf((ow + 1) * W / (float)out_w);
        if (w_end[ow] <= w_start[ow])
            w_end[ow] = w_start[ow] + 1;
    }

    int out_coords[4] = {0};
    while (1)
    {
        int n = out_coords[0];
        int c = out_coords[1];
        int oh = out_coords[2];
        int ow = out_coords[3];

        int start_h = h_start[oh];
        int end_h = h_end[oh];
        int start_w = w_start[ow];
        int end_w = w_end[ow];
        int kernel_h = end_h - start_h;
        int kernel_w = end_w - start_w;
        float inv_count = 1.0f / (kernel_h * kernel_w);

        double sum = 0.0;
        for (int ih = start_h; ih < end_h; ++ih)
        {
            for (int iw = start_w; iw < end_w; ++iw)
            {
                int in_coords[4] = {n, c, ih, iw};
                ptrdiff_t in_off = util_offset_from_coords(in_coords, x_strides, 4);
                sum += x->data[in_off];
            }
        }

        ptrdiff_t out_off = util_offset_from_coords(out_coords, out_strides, 4);
        out->data[out_off] = (float)(sum * inv_count);

        if (util_increment_coords(out_coords, out->dims, 4))
            break;
    }

    free(h_start);
    free(h_end);
    free(w_start);
    free(w_end);
    return TENSOR_OK;
}

/* ==================== 嵌入层 ==================== */
TensorStatus tensor_embedding(const Tensor *input, const Tensor *weight,
                              int padding_idx, Tensor *out)
{
    if (!input || !weight || !out)
        return TENSOR_ERR_NULL_PTR;
    if (weight->ndim != 2)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int vocab_size = weight->dims[0];
    int emb_dim = weight->dims[1];
    int in_ndim = input->ndim;

    // 输出形状 = input.shape + [emb_dim]
    int out_ndim = in_ndim + 1;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < in_ndim; ++i)
        out_dims[i] = input->dims[i];
    out_dims[in_ndim] = emb_dim;

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; ++i)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int in_strides[TENSOR_MAX_DIM], w_strides[2], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(input, in_strides);
    util_get_effective_strides(weight, w_strides);
    util_get_effective_strides(out, out_strides);

    int in_coords[TENSOR_MAX_DIM] = {0};
    while (1)
    {
        ptrdiff_t in_off = util_offset_from_coords(in_coords, in_strides, in_ndim);
        float idx_float = input->data[in_off];
        TensorStatus st;
        int idx = tensor_float_to_index(idx_float, vocab_size, &st);
        if (st != TENSOR_OK)
            return st;

        ptrdiff_t out_base = 0;
        for (int i = 0; i < in_ndim; ++i)
            out_base += in_coords[i] * out_strides[i];

        if (padding_idx >= 0 && idx == padding_idx)
        {
            for (int e = 0; e < emb_dim; ++e)
            {
                ptrdiff_t out_off = out_base + e * out_strides[in_ndim];
                out->data[out_off] = 0.0f;
            }
        }
        else
        {
            ptrdiff_t w_base = idx * w_strides[0];
            for (int e = 0; e < emb_dim; ++e)
            {
                ptrdiff_t out_off = out_base + e * out_strides[in_ndim];
                ptrdiff_t w_off = w_base + e * w_strides[1];
                out->data[out_off] = weight->data[w_off];
            }
        }

        if (util_increment_coords(in_coords, input->dims, in_ndim))
            break;
    }
    return TENSOR_OK;
}

/* ==================== 1D 上采样辅助函数 ==================== */
static float cubic_interp1d(float p0, float p1, float p2, float p3, float t)
{
    return 0.5f * ((-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t +
                   (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                   (-p0 + p2) * t +
                   2 * p1);
}

static TensorStatus upsample1d_linear(const Tensor *x, int scale, int align_corners, Tensor *out)
{
    int N = x->dims[0], C = x->dims[1], L = x->dims[2];
    int out_len = out->dims[2];
    int x_strides[3], out_strides[3];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    float inv_scale = 1.0f / scale;
    float factor = align_corners ? (L - 1.0f) / (out_len - 1.0f) : inv_scale;

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int ol = 0; ol < out_len; ++ol)
            {
                float xf;
                if (align_corners)
                {
                    if (ol == 0)
                        xf = 0.0f;
                    else if (ol == out_len - 1)
                        xf = L - 1.0f;
                    else
                        xf = ol * factor;
                }
                else
                {
                    xf = (ol + 0.5f) * inv_scale - 0.5f;
                }

                int i0 = (int)floorf(xf);
                int i1 = i0 + 1;
                float t = xf - i0;

                if (i0 < 0)
                {
                    i0 = 0;
                    i1 = 0;
                    t = 0.0f;
                }
                else if (i1 >= L)
                {
                    i0 = L - 1;
                    i1 = L - 1;
                    t = 0.0f;
                }

                ptrdiff_t off0 = n * x_strides[0] + c * x_strides[1] + i0 * x_strides[2];
                ptrdiff_t off1 = n * x_strides[0] + c * x_strides[1] + i1 * x_strides[2];
                float v0 = x->data[off0];
                float v1 = x->data[off1];
                float val = v0 * (1 - t) + v1 * t;

                ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] + ol * out_strides[2];
                out->data[out_off] = val;
            }
        }
    }
    return TENSOR_OK;
}

static TensorStatus upsample1d_cubic(const Tensor *x, int scale, int align_corners, Tensor *out)
{
    int N = x->dims[0], C = x->dims[1], L = x->dims[2];
    int out_len = out->dims[2];
    int x_strides[3], out_strides[3];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    float inv_scale = 1.0f / scale;
    float factor = align_corners ? (L - 1.0f) / (out_len - 1.0f) : inv_scale;

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int ol = 0; ol < out_len; ++ol)
            {
                float xf;
                if (align_corners)
                {
                    if (ol == 0)
                        xf = 0.0f;
                    else if (ol == out_len - 1)
                        xf = L - 1.0f;
                    else
                        xf = ol * factor;
                }
                else
                {
                    xf = (ol + 0.5f) * inv_scale - 0.5f;
                }

                int i0 = (int)floorf(xf) - 1;
                int i1 = i0 + 1;
                int i2 = i0 + 2;
                int i3 = i0 + 3;
                float t = xf - floorf(xf);

                i0 = reflect_coord(i0, L);
                i1 = reflect_coord(i1, L);
                i2 = reflect_coord(i2, L);
                i3 = reflect_coord(i3, L);

                ptrdiff_t off0 = n * x_strides[0] + c * x_strides[1] + i0 * x_strides[2];
                ptrdiff_t off1 = n * x_strides[0] + c * x_strides[1] + i1 * x_strides[2];
                ptrdiff_t off2 = n * x_strides[0] + c * x_strides[1] + i2 * x_strides[2];
                ptrdiff_t off3 = n * x_strides[0] + c * x_strides[1] + i3 * x_strides[2];
                float p0 = x->data[off0];
                float p1 = x->data[off1];
                float p2 = x->data[off2];
                float p3 = x->data[off3];
                float val = cubic_interp1d(p0, p1, p2, p3, t);

                ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] + ol * out_strides[2];
                out->data[out_off] = val;
            }
        }
    }
    return TENSOR_OK;
}

/* ==================== 1D 上采样 API ==================== */
TensorStatus tensor_upsample1d(const Tensor *x, int scale,
                               InterpMode mode, int align_corners,
                               Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 3)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (scale <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    int N = x->dims[0], C = x->dims[1], L = x->dims[2];
    int out_len = L * scale;

    if (out->ndim != 3 || out->dims[0] != N || out->dims[1] != C || out->dims[2] != out_len)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    if (mode == UPSAMPLE_NEAREST)
    {
        int x_strides[3], out_strides[3];
        util_get_effective_strides(x, x_strides);
        util_get_effective_strides(out, out_strides);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int ol = 0; ol < out_len; ++ol)
                {
                    int il = ol / scale;
                    ptrdiff_t x_off = n * x_strides[0] + c * x_strides[1] + il * x_strides[2];
                    ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] + ol * out_strides[2];
                    out->data[out_off] = x->data[x_off];
                }
        return TENSOR_OK;
    }
    else if (mode == UPSAMPLE_LINEAR)
    {
        return upsample1d_linear(x, scale, align_corners, out);
    }
    else if (mode == UPSAMPLE_CUBIC)
    {
        return upsample1d_cubic(x, scale, align_corners, out);
    }
    return TENSOR_ERR_INVALID_PARAM;
}

/* ==================== 3D 上采样辅助函数 ==================== */
static TensorStatus upsample3d_linear(const Tensor *x,
                                      int scale_d, int scale_h, int scale_w,
                                      int align_corners, Tensor *out)
{
    int N = x->dims[0], C = x->dims[1], D = x->dims[2], H = x->dims[3], W = x->dims[4];
    int out_d = out->dims[2], out_h = out->dims[3], out_w = out->dims[4];

    int x_strides[5], out_strides[5];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    float inv_scale_d = 1.0f / scale_d;
    float inv_scale_h = 1.0f / scale_h;
    float inv_scale_w = 1.0f / scale_w;
    float factor_d = align_corners ? (D - 1.0f) / (out_d - 1.0f) : inv_scale_d;
    float factor_h = align_corners ? (H - 1.0f) / (out_h - 1.0f) : inv_scale_h;
    float factor_w = align_corners ? (W - 1.0f) / (out_w - 1.0f) : inv_scale_w;

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int od = 0; od < out_d; ++od)
            {
                float df;
                if (align_corners)
                {
                    if (od == 0)
                        df = 0.0f;
                    else if (od == out_d - 1)
                        df = D - 1.0f;
                    else
                        df = od * factor_d;
                }
                else
                {
                    df = (od + 0.5f) * inv_scale_d - 0.5f;
                }
                int id0 = (int)floorf(df);
                int id1 = id0 + 1;
                float td = df - id0;
                if (id0 < 0)
                {
                    id0 = 0;
                    id1 = 0;
                    td = 0.0f;
                }
                else if (id1 >= D)
                {
                    id0 = D - 1;
                    id1 = D - 1;
                    td = 0.0f;
                }

                for (int oh = 0; oh < out_h; ++oh)
                {
                    float hf;
                    if (align_corners)
                    {
                        if (oh == 0)
                            hf = 0.0f;
                        else if (oh == out_h - 1)
                            hf = H - 1.0f;
                        else
                            hf = oh * factor_h;
                    }
                    else
                    {
                        hf = (oh + 0.5f) * inv_scale_h - 0.5f;
                    }
                    int ih0 = (int)floorf(hf);
                    int ih1 = ih0 + 1;
                    float th = hf - ih0;
                    if (ih0 < 0)
                    {
                        ih0 = 0;
                        ih1 = 0;
                        th = 0.0f;
                    }
                    else if (ih1 >= H)
                    {
                        ih0 = H - 1;
                        ih1 = H - 1;
                        th = 0.0f;
                    }

                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        float wf;
                        if (align_corners)
                        {
                            if (ow == 0)
                                wf = 0.0f;
                            else if (ow == out_w - 1)
                                wf = W - 1.0f;
                            else
                                wf = ow * factor_w;
                        }
                        else
                        {
                            wf = (ow + 0.5f) * inv_scale_w - 0.5f;
                        }
                        int iw0 = (int)floorf(wf);
                        int iw1 = iw0 + 1;
                        float tw = wf - iw0;
                        if (iw0 < 0)
                        {
                            iw0 = 0;
                            iw1 = 0;
                            tw = 0.0f;
                        }
                        else if (iw1 >= W)
                        {
                            iw0 = W - 1;
                            iw1 = W - 1;
                            tw = 0.0f;
                        }

                        // 获取8个点
#define GET_VAL(d, h, w) x->data[n * x_strides[0] + c * x_strides[1] + \
                                 (d) * x_strides[2] + (h) * x_strides[3] + (w) * x_strides[4]]
                        float v000 = GET_VAL(id0, ih0, iw0);
                        float v001 = GET_VAL(id0, ih0, iw1);
                        float v010 = GET_VAL(id0, ih1, iw0);
                        float v011 = GET_VAL(id0, ih1, iw1);
                        float v100 = GET_VAL(id1, ih0, iw0);
                        float v101 = GET_VAL(id1, ih0, iw1);
                        float v110 = GET_VAL(id1, ih1, iw0);
                        float v111 = GET_VAL(id1, ih1, iw1);
#undef GET_VAL

                        // 三线性插值
                        float v00 = v000 * (1 - tw) + v001 * tw;
                        float v01 = v010 * (1 - tw) + v011 * tw;
                        float v10 = v100 * (1 - tw) + v101 * tw;
                        float v11 = v110 * (1 - tw) + v111 * tw;

                        float v0 = v00 * (1 - th) + v01 * th;
                        float v1 = v10 * (1 - th) + v11 * th;

                        float val = v0 * (1 - td) + v1 * td;

                        ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] +
                                            od * out_strides[2] + oh * out_strides[3] + ow * out_strides[4];
                        out->data[out_off] = val;
                    }
                }
            }
        }
    }
    return TENSOR_OK;
}

static TensorStatus upsample3d_cubic(const Tensor *x,
                                     int scale_d, int scale_h, int scale_w,
                                     int align_corners, Tensor *out)
{
    int N = x->dims[0], C = x->dims[1], D = x->dims[2], H = x->dims[3], W = x->dims[4];
    int out_d = out->dims[2], out_h = out->dims[3], out_w = out->dims[4];

    int x_strides[5], out_strides[5];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    float inv_scale_d = 1.0f / scale_d;
    float inv_scale_h = 1.0f / scale_h;
    float inv_scale_w = 1.0f / scale_w;
    float factor_d = align_corners ? (D - 1.0f) / (out_d - 1.0f) : inv_scale_d;
    float factor_h = align_corners ? (H - 1.0f) / (out_h - 1.0f) : inv_scale_h;
    float factor_w = align_corners ? (W - 1.0f) / (out_w - 1.0f) : inv_scale_w;

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int od = 0; od < out_d; ++od)
            {
                float df;
                if (align_corners)
                {
                    if (od == 0)
                        df = 0.0f;
                    else if (od == out_d - 1)
                        df = D - 1.0f;
                    else
                        df = od * factor_d;
                }
                else
                {
                    df = (od + 0.5f) * inv_scale_d - 0.5f;
                }
                int id0 = (int)floorf(df) - 1;
                int id1 = id0 + 1;
                int id2 = id0 + 2;
                int id3 = id0 + 3;
                float td = df - floorf(df);

                id0 = reflect_coord(id0, D);
                id1 = reflect_coord(id1, D);
                id2 = reflect_coord(id2, D);
                id3 = reflect_coord(id3, D);

                for (int oh = 0; oh < out_h; ++oh)
                {
                    float hf;
                    if (align_corners)
                    {
                        if (oh == 0)
                            hf = 0.0f;
                        else if (oh == out_h - 1)
                            hf = H - 1.0f;
                        else
                            hf = oh * factor_h;
                    }
                    else
                    {
                        hf = (oh + 0.5f) * inv_scale_h - 0.5f;
                    }
                    int ih0 = (int)floorf(hf) - 1;
                    int ih1 = ih0 + 1;
                    int ih2 = ih0 + 2;
                    int ih3 = ih0 + 3;
                    float th = hf - floorf(hf);

                    ih0 = reflect_coord(ih0, H);
                    ih1 = reflect_coord(ih1, H);
                    ih2 = reflect_coord(ih2, H);
                    ih3 = reflect_coord(ih3, H);

                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        float wf;
                        if (align_corners)
                        {
                            if (ow == 0)
                                wf = 0.0f;
                            else if (ow == out_w - 1)
                                wf = W - 1.0f;
                            else
                                wf = ow * factor_w;
                        }
                        else
                        {
                            wf = (ow + 0.5f) * inv_scale_w - 0.5f;
                        }
                        int iw0 = (int)floorf(wf) - 1;
                        int iw1 = iw0 + 1;
                        int iw2 = iw0 + 2;
                        int iw3 = iw0 + 3;
                        float tw = wf - floorf(wf);

                        iw0 = reflect_coord(iw0, W);
                        iw1 = reflect_coord(iw1, W);
                        iw2 = reflect_coord(iw2, W);
                        iw3 = reflect_coord(iw3, W);

                        // 获取64个点，并在三个方向上进行三次插值
                        // 首先在 w 方向插值，得到 4x4 的中间值
                        float temp[4][4];
                        int d_idx[4] = {id0, id1, id2, id3};
                        int h_idx[4] = {ih0, ih1, ih2, ih3};
                        int w_idx[4] = {iw0, iw1, iw2, iw3};

                        for (int di = 0; di < 4; ++di)
                        {
                            for (int hi = 0; hi < 4; ++hi)
                            {
                                float p[4];
                                for (int wi = 0; wi < 4; ++wi)
                                {
                                    ptrdiff_t off = n * x_strides[0] + c * x_strides[1] +
                                                    d_idx[di] * x_strides[2] +
                                                    h_idx[hi] * x_strides[3] +
                                                    w_idx[wi] * x_strides[4];
                                    p[wi] = x->data[off];
                                }
                                temp[di][hi] = cubic_interp1d(p[0], p[1], p[2], p[3], tw);
                            }
                        }

                        // 在 h 方向插值
                        float temp2[4];
                        for (int di = 0; di < 4; ++di)
                        {
                            temp2[di] = cubic_interp1d(temp[di][0], temp[di][1], temp[di][2], temp[di][3], th);
                        }

                        // 在 d 方向插值
                        float val = cubic_interp1d(temp2[0], temp2[1], temp2[2], temp2[3], td);

                        ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] +
                                            od * out_strides[2] + oh * out_strides[3] + ow * out_strides[4];
                        out->data[out_off] = val;
                    }
                }
            }
        }
    }
    return TENSOR_OK;
}

/* ==================== 3D 上采样 API ==================== */
TensorStatus tensor_upsample3d(const Tensor *x,
                               int scale_d, int scale_h, int scale_w,
                               InterpMode mode, int align_corners,
                               Tensor *out)
{
    if (!x || !out)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 5)
        return TENSOR_ERR_SHAPE_MISMATCH;
    if (scale_d <= 0 || scale_h <= 0 || scale_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    int N = x->dims[0], C = x->dims[1], D = x->dims[2], H = x->dims[3], W = x->dims[4];
    int out_d = D * scale_d, out_h = H * scale_h, out_w = W * scale_w;

    if (out->ndim != 5 ||
        out->dims[0] != N || out->dims[1] != C ||
        out->dims[2] != out_d || out->dims[3] != out_h || out->dims[4] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    if (mode == UPSAMPLE_NEAREST)
    {
        int x_strides[5], out_strides[5];
        util_get_effective_strides(x, x_strides);
        util_get_effective_strides(out, out_strides);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int od = 0; od < out_d; ++od)
                {
                    int id = od / scale_d;
                    for (int oh = 0; oh < out_h; ++oh)
                    {
                        int ih = oh / scale_h;
                        for (int ow = 0; ow < out_w; ++ow)
                        {
                            int iw = ow / scale_w;
                            ptrdiff_t x_off = n * x_strides[0] + c * x_strides[1] +
                                              id * x_strides[2] + ih * x_strides[3] + iw * x_strides[4];
                            ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] +
                                                od * out_strides[2] + oh * out_strides[3] + ow * out_strides[4];
                            out->data[out_off] = x->data[x_off];
                        }
                    }
                }
        return TENSOR_OK;
    }
    else if (mode == UPSAMPLE_LINEAR)
    {
        return upsample3d_linear(x, scale_d, scale_h, scale_w, align_corners, out);
    }
    else if (mode == UPSAMPLE_CUBIC)
    {
        return upsample3d_cubic(x, scale_d, scale_h, scale_w, align_corners, out);
    }
    return TENSOR_ERR_INVALID_PARAM;
}

/* ==================== 1D 自适应平均池化 ==================== */
TensorStatus tensor_adaptive_avg_pool1d(const Tensor *x,
                                        const int *output_size,
                                        Tensor *out)
{
    if (!x || !out || !output_size)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 3)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int N = x->dims[0], C = x->dims[1], L = x->dims[2];
    int out_len = output_size[0];
    if (out_len <= 0 || out_len > L)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 3 || out->dims[0] != N || out->dims[1] != C || out->dims[2] != out_len)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[3], out_strides[3];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int *start = (int *)malloc(out_len * sizeof(int));
    int *end = (int *)malloc(out_len * sizeof(int));
    if (!start || !end)
    {
        free(start);
        free(end);
        return TENSOR_ERR_MEMORY;
    }

    for (int ol = 0; ol < out_len; ++ol)
    {
        start[ol] = (int)floorf(ol * L / (float)out_len);
        end[ol] = (int)floorf((ol + 1) * L / (float)out_len);
        if (end[ol] <= start[ol])
            end[ol] = start[ol] + 1;
    }

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int ol = 0; ol < out_len; ++ol)
            {
                double sum = 0.0;
                int cnt = 0;
                for (int il = start[ol]; il < end[ol]; ++il)
                {
                    ptrdiff_t x_off = n * x_strides[0] + c * x_strides[1] + il * x_strides[2];
                    sum += x->data[x_off];
                    cnt++;
                }
                float avg = (cnt == 0) ? 0.0f : (float)(sum / cnt);
                ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] + ol * out_strides[2];
                out->data[out_off] = avg;
            }
        }
    }

    free(start);
    free(end);
    return TENSOR_OK;
}

/* ==================== 3D 自适应平均池化 ==================== */
TensorStatus tensor_adaptive_avg_pool3d(const Tensor *x,
                                        const int *output_size,
                                        Tensor *out)
{
    if (!x || !out || !output_size)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 5)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int N = x->dims[0], C = x->dims[1], D = x->dims[2], H = x->dims[3], W = x->dims[4];
    int out_d = output_size[0], out_h = output_size[1], out_w = output_size[2];
    if (out_d <= 0 || out_d > D || out_h <= 0 || out_h > H || out_w <= 0 || out_w > W)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 5 ||
        out->dims[0] != N || out->dims[1] != C ||
        out->dims[2] != out_d || out->dims[3] != out_h || out->dims[4] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[5], out_strides[5];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(out, out_strides);

    int *d_start = (int *)malloc(out_d * sizeof(int));
    int *d_end = (int *)malloc(out_d * sizeof(int));
    int *h_start = (int *)malloc(out_h * sizeof(int));
    int *h_end = (int *)malloc(out_h * sizeof(int));
    int *w_start = (int *)malloc(out_w * sizeof(int));
    int *w_end = (int *)malloc(out_w * sizeof(int));
    if (!d_start || !d_end || !h_start || !h_end || !w_start || !w_end)
    {
        free(d_start);
        free(d_end);
        free(h_start);
        free(h_end);
        free(w_start);
        free(w_end);
        return TENSOR_ERR_MEMORY;
    }

    for (int od = 0; od < out_d; ++od)
    {
        d_start[od] = (int)floorf(od * D / (float)out_d);
        d_end[od] = (int)floorf((od + 1) * D / (float)out_d);
        if (d_end[od] <= d_start[od])
            d_end[od] = d_start[od] + 1;
    }
    for (int oh = 0; oh < out_h; ++oh)
    {
        h_start[oh] = (int)floorf(oh * H / (float)out_h);
        h_end[oh] = (int)floorf((oh + 1) * H / (float)out_h);
        if (h_end[oh] <= h_start[oh])
            h_end[oh] = h_start[oh] + 1;
    }
    for (int ow = 0; ow < out_w; ++ow)
    {
        w_start[ow] = (int)floorf(ow * W / (float)out_w);
        w_end[ow] = (int)floorf((ow + 1) * W / (float)out_w);
        if (w_end[ow] <= w_start[ow])
            w_end[ow] = w_start[ow] + 1;
    }

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int od = 0; od < out_d; ++od)
            {
                for (int oh = 0; oh < out_h; ++oh)
                {
                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        double sum = 0.0;
                        int cnt = 0;
                        for (int id = d_start[od]; id < d_end[od]; ++id)
                            for (int ih = h_start[oh]; ih < h_end[oh]; ++ih)
                                for (int iw = w_start[ow]; iw < w_end[ow]; ++iw)
                                {
                                    ptrdiff_t x_off = n * x_strides[0] + c * x_strides[1] +
                                                      id * x_strides[2] + ih * x_strides[3] + iw * x_strides[4];
                                    sum += x->data[x_off];
                                    cnt++;
                                }
                        float avg = (cnt == 0) ? 0.0f : (float)(sum / cnt);
                        ptrdiff_t out_off = n * out_strides[0] + c * out_strides[1] +
                                            od * out_strides[2] + oh * out_strides[3] + ow * out_strides[4];
                        out->data[out_off] = avg;
                    }
                }
            }
        }
    }

    free(d_start);
    free(d_end);
    free(h_start);
    free(h_end);
    free(w_start);
    free(w_end);
    return TENSOR_OK;
}

/* ==================== 1D 最大反池化 ==================== */
TensorStatus tensor_max_unpool1d(const Tensor *x, const Tensor *indices,
                                 const int *output_size, Tensor *out)
{
    if (!x || !indices || !out || !output_size)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 3 || indices->ndim != 3)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < 3; ++i)
        if (x->dims[i] != indices->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    int N = x->dims[0], C = x->dims[1], L_in = x->dims[2];
    int out_len = output_size[0];
    if (out_len <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 3 || out->dims[0] != N || out->dims[1] != C || out->dims[2] != out_len)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[3], idx_strides[3], out_strides[3];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(indices, idx_strides);
    util_get_effective_strides(out, out_strides);

    util_clear_tensor(out);

    int coords[3] = {0};
    while (1)
    {
        ptrdiff_t x_off = util_offset_from_coords(coords, x_strides, 3);
        ptrdiff_t idx_off = util_offset_from_coords(coords, idx_strides, 3);
        float val = x->data[x_off];
        float idx_float = indices->data[idx_off];
        TensorStatus st;
        int idx_val = tensor_float_to_index(idx_float, out_len, &st);
        if (st != TENSOR_OK)
            return st;

        int out_coords[3] = {coords[0], coords[1], idx_val};
        ptrdiff_t out_off = util_offset_from_coords(out_coords, out_strides, 3);
        out->data[out_off] += val;

        if (util_increment_coords(coords, x->dims, 3))
            break;
    }
    return TENSOR_OK;
}

/* ==================== 3D 最大反池化 ==================== */
TensorStatus tensor_max_unpool3d(const Tensor *x, const Tensor *indices,
                                 const int *output_size, Tensor *out)
{
    if (!x || !indices || !out || !output_size)
        return TENSOR_ERR_NULL_PTR;
    if (x->ndim != 5 || indices->ndim != 5)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < 5; ++i)
        if (x->dims[i] != indices->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    int N = x->dims[0], C = x->dims[1];
    int D_in = x->dims[2], H_in = x->dims[3], W_in = x->dims[4];
    int out_d = output_size[0], out_h = output_size[1], out_w = output_size[2];
    if (out_d <= 0 || out_h <= 0 || out_w <= 0)
        return TENSOR_ERR_INVALID_PARAM;

    if (out->ndim != 5 ||
        out->dims[0] != N || out->dims[1] != C ||
        out->dims[2] != out_d || out->dims[3] != out_h || out->dims[4] != out_w)
        return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int x_strides[5], idx_strides[5], out_strides[5];
    util_get_effective_strides(x, x_strides);
    util_get_effective_strides(indices, idx_strides);
    util_get_effective_strides(out, out_strides);

    util_clear_tensor(out);

    int coords[5] = {0};
    while (1)
    {
        ptrdiff_t x_off = util_offset_from_coords(coords, x_strides, 5);
        ptrdiff_t idx_off = util_offset_from_coords(coords, idx_strides, 5);
        float val = x->data[x_off];
        float idx_float = indices->data[idx_off];
        TensorStatus st;
        int idx_val = tensor_float_to_index(idx_float, out_d * out_h * out_w, &st);
        if (st != TENSOR_OK)
            return st;

        int out_coords[5] = {
            coords[0], coords[1],
            idx_val / (out_h * out_w),
            (idx_val / out_w) % out_h,
            idx_val % out_w};
        ptrdiff_t out_off = util_offset_from_coords(out_coords, out_strides, 5);
        out->data[out_off] += val;

        if (util_increment_coords(coords, x->dims, 5))
            break;
    }
    return TENSOR_OK;
}