#ifndef TENSOR_NN_OPS_H
#define TENSOR_NN_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file nn_ops.h
     * @brief 神经网络层：卷积、池化、归一化、激活函数等
     */

    /* ==================== 卷积参数 ==================== */

    typedef struct
    {
        int pad[3];      // 每个维度的填充（前后各一，顺序为高度、宽度、深度）
        int stride[3];   // 步长
        int dilation[3]; // 扩张率
        int groups;      // 分组数
    } ConvParams;

    /* ==================== 卷积 ==================== */

    TensorStatus tensor_conv1d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               ConvParams params, Tensor *output);
    TensorStatus tensor_conv2d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               ConvParams params, Tensor *output);
    TensorStatus tensor_conv3d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               ConvParams params, Tensor *output);
    TensorStatus tensor_conv_transpose1d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                                         ConvParams params, Tensor *output);
    TensorStatus tensor_conv_transpose2d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                                         ConvParams params, Tensor *output);
    TensorStatus tensor_conv_transpose3d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                                         ConvParams params, Tensor *output);

    /* ==================== 池化参数 ==================== */

    typedef enum
    {
        POOL_MAX, // 最大池化
        POOL_AVG, // 平均池化
        POOL_L2   // L2 池化
    } PoolType;

    typedef struct
    {
        int kernel[3];         // 核大小
        int pad[3];            // 填充
        int stride[3];         // 步长
        int ceil_mode;         // 使用 ceil 计算输出尺寸（非零）
        int count_include_pad; // 平均池化是否包含填充区域（非零）
    } PoolParams;

    /* ==================== 池化 ==================== */

    TensorStatus tensor_pool1d(const Tensor *input, PoolType type, PoolParams params, Tensor *output);
    TensorStatus tensor_pool2d(const Tensor *input, PoolType type, PoolParams params, Tensor *output);
    TensorStatus tensor_pool3d(const Tensor *input, PoolType type, PoolParams params, Tensor *output);
    TensorStatus tensor_global_avg_pool2d(const Tensor *input, Tensor *output);
    TensorStatus tensor_global_max_pool2d(const Tensor *input, Tensor *output);

    /* ==================== 归一化 ==================== */

    /**
     * @brief 批归一化（推理模式）
     * @param x 输入张量 [N, C, ...]
     * @param mean 均值 [C]
     * @param var 方差 [C]
     * @param scale 缩放 [C] (可为 NULL)
     * @param bias 偏置 [C] (可为 NULL)
     * @param epsilon 小常数
     * @param y 输出张量，形状同 x
     */
    TensorStatus tensor_batchnorm(const Tensor *x, const Tensor *mean, const Tensor *var,
                                  const Tensor *scale, const Tensor *bias,
                                  float epsilon, Tensor *y);

    /**
     * @brief 层归一化（对最后一维）
     * @param x 输入张量
     * @param scale 缩放，形状同最后一维 (可为 NULL)
     * @param bias 偏置，形状同最后一维 (可为 NULL)
     * @param epsilon 小常数
     * @param y 输出张量，形状同 x
     */
    TensorStatus tensor_layernorm(const Tensor *x, const Tensor *scale, const Tensor *bias,
                                  float epsilon, Tensor *y);

    /**
     * @brief 实例归一化（对每个样本每个通道）
     * @param x 输入张量 [N, C, H, W] 或 [N, C, D, H, W]
     * @param scale 缩放 [C] (可为 NULL)
     * @param bias 偏置 [C] (可为 NULL)
     * @param epsilon 小常数
     * @param y 输出张量，形状同 x
     */
    TensorStatus tensor_instancenorm(const Tensor *x, const Tensor *scale, const Tensor *bias,
                                     float epsilon, Tensor *y);

    /**
     * @brief 组归一化
     * @param x 输入张量 [N, C, ...]
     * @param scale 缩放 [C] (可为 NULL)
     * @param bias 偏置 [C] (可为 NULL)
     * @param num_groups 分组数
     * @param epsilon 小常数
     * @param y 输出张量，形状同 x
     */
    TensorStatus tensor_groupnorm(const Tensor *x, const Tensor *scale, const Tensor *bias,
                                  int num_groups, float epsilon, Tensor *y);

    /**
     * @brief 本地响应归一化 (LRN)
     * @param x 输入张量 [N, C, H, W]
     * @param size 邻域大小（必须为奇数）
     * @param alpha 缩放因子
     * @param beta 指数
     * @param bias 偏置
     * @param y 输出张量，形状同 x
     */
    TensorStatus tensor_lrn(const Tensor *x, int size, float alpha, float beta, float bias, Tensor *y);

    /* ==================== 激活函数 ==================== */

    TensorStatus tensor_relu(const Tensor *x, Tensor *out);
    TensorStatus tensor_leaky_relu(const Tensor *x, float alpha, Tensor *out);
    TensorStatus tensor_elu(const Tensor *x, float alpha, Tensor *out);
    TensorStatus tensor_selu(const Tensor *x, float alpha, float scale, Tensor *out);
    TensorStatus tensor_gelu(const Tensor *x, Tensor *out);                       // 高斯误差线性单元（近似）
    TensorStatus tensor_swish(const Tensor *x, Tensor *out);                      // x * sigmoid(x)
    TensorStatus tensor_mish(const Tensor *x, Tensor *out);                       // x * tanh(softplus(x))
    TensorStatus tensor_softplus(const Tensor *x, Tensor *out);                   // log(1+exp(x))
    TensorStatus tensor_softsign(const Tensor *x, Tensor *out);                   // x / (1+|x|)
    TensorStatus tensor_hardswish(const Tensor *x, Tensor *out);                  // x * clamp(x+3,0,6)/6
    TensorStatus tensor_hardsigmoid(const Tensor *x, Tensor *out);                // clamp(x+3,0,6)/6
    TensorStatus tensor_prelu(const Tensor *x, const Tensor *alpha, Tensor *out); // 参数化ReLU

    /* ==================== 其他层 ==================== */

    /**
     * @brief 全连接层: y = input @ weight^T + bias
     * @param input 输入张量，最后一维大小为 in_features
     * @param weight 权重张量 [out_features, in_features]
     * @param bias 偏置张量 [out_features] (可为 NULL)
     * @param output 输出张量
     */
    TensorStatus tensor_linear(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               Tensor *output);

    /**
     * @brief Dropout（训练模式）
     * @param x 输入张量
     * @param p 丢弃概率 (0 <= p < 1)
     * @param training 非零表示训练模式，零表示推理模式（直接返回 x）
     * @param out 输出张量
     */
    TensorStatus tensor_dropout(const Tensor *x, float p, int training, Tensor *out);

    /**
     * @brief Softmax
     * @param x 输入张量
     * @param axis 应用 softmax 的轴
     * @param out 输出张量
     */
    TensorStatus tensor_softmax(const Tensor *x, int axis, Tensor *out);

    /**
     * @brief LogSoftmax
     * @param x 输入张量
     * @param axis 轴
     * @param out 输出张量
     */
    TensorStatus tensor_log_softmax(const Tensor *x, int axis, Tensor *out);

    typedef enum
    {
        UPSAMPLE_NEAREST, // 最近邻插值
        UPSAMPLE_LINEAR,  // 线性插值（双线性/三线性）
        UPSAMPLE_CUBIC    // 三次插值
    } InterpMode;

    /**
     * @brief 2D 上采样
     * @param x 输入张量 [N, C, H, W]
     * @param scale_h 高度缩放因子
     * @param scale_w 宽度缩放因子
     * @param mode 插值模式
     * @param align_corners 是否对角线对齐（UPSAMPLE_LINEAR/UPSAMPLE_CUBIC时）
     * @param out 输出张量
     */
    TensorStatus tensor_upsample2d(const Tensor *x, int scale_h, int scale_w,
                                   InterpMode mode, int align_corners, Tensor *out);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_NN_OPS_H