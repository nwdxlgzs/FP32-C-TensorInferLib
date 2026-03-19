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

    /**
     * @brief 卷积操作参数结构体
     */
    typedef struct
    {
        int pad[3];      //!< 每个维度的填充（前后各一，顺序为高度、宽度、深度）
        int stride[3];   //!< 步长
        int dilation[3]; //!< 扩张率
        int groups;      //!< 分组数
    } ConvParams;

    /* ==================== 卷积函数 ==================== */

    /**
     * @brief 一维卷积
     * @param input  输入张量，形状一般为 [N, C, L] 或 [C, L]
     * @param weight 卷积核，形状为 [out_channels, in_channels/groups, kernel_size]
     * @param bias   偏置张量，形状为 [out_channels]，可为 NULL
     * @param params 卷积参数
     * @param output 输出张量，形状根据参数计算得出
     * @return TensorStatus
     */
    TensorStatus tensor_conv1d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               ConvParams params, Tensor *output);

    /**
     * @brief 二维卷积
     * @param input  输入张量，形状一般为 [N, C, H, W] 或 [C, H, W]
     * @param weight 卷积核，形状为 [out_channels, in_channels/groups, kernel_h, kernel_w]
     * @param bias   偏置张量，形状为 [out_channels]，可为 NULL
     * @param params 卷积参数
     * @param output 输出张量，形状根据参数计算得出
     * @return TensorStatus
     */
    TensorStatus tensor_conv2d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               ConvParams params, Tensor *output);

    /**
     * @brief 三维卷积
     * @param input  输入张量，形状一般为 [N, C, D, H, W] 或 [C, D, H, W]
     * @param weight 卷积核，形状为 [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
     * @param bias   偏置张量，形状为 [out_channels]，可为 NULL
     * @param params 卷积参数
     * @param output 输出张量，形状根据参数计算得出
     * @return TensorStatus
     */
    TensorStatus tensor_conv3d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               ConvParams params, Tensor *output);

    /**
     * @brief 一维转置卷积（反卷积）
     * @param input  输入张量
     * @param weight 卷积核
     * @param bias   偏置（可为 NULL）
     * @param params 卷积参数
     * @param output 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_conv_transpose1d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                                         ConvParams params, Tensor *output);

    /**
     * @brief 二维转置卷积
     */
    TensorStatus tensor_conv_transpose2d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                                         ConvParams params, Tensor *output);

    /**
     * @brief 三维转置卷积
     */
    TensorStatus tensor_conv_transpose3d(const Tensor *input, const Tensor *weight, const Tensor *bias,
                                         ConvParams params, Tensor *output);

    /* ==================== 池化参数 ==================== */

    /**
     * @brief 池化类型枚举
     */
    typedef enum
    {
        POOL_MAX, //!< 最大池化
        POOL_AVG, //!< 平均池化
        POOL_L2   //!< L2 池化（计算窗口内平方和的平方根）
    } PoolType;

    /**
     * @brief 池化操作参数结构体
     */
    typedef struct
    {
        int kernel[3];         //!< 核大小，每个维度的窗口大小
        int pad[3];            //!< 填充
        int stride[3];         //!< 步长
        int ceil_mode;         //!< 使用 ceil 计算输出尺寸（非零表示使用 ceil，否则使用 floor）
        int count_include_pad; //!< 平均池化是否包含填充区域（非零表示包含）
    } PoolParams;

    /* ==================== 池化函数 ==================== */

    /**
     * @brief 一维池化
     * @param input  输入张量
     * @param type   池化类型
     * @param params 池化参数
     * @param output 输出张量
     * @return TensorStatus
     */
    TensorStatus tensor_pool1d(const Tensor *input, PoolType type, PoolParams params, Tensor *output);

    /**
     * @brief 二维池化
     */
    TensorStatus tensor_pool2d(const Tensor *input, PoolType type, PoolParams params, Tensor *output);

    /**
     * @brief 三维池化
     */
    TensorStatus tensor_pool3d(const Tensor *input, PoolType type, PoolParams params, Tensor *output);

    /**
     * @brief 二维全局平均池化
     * @param input  输入张量，形状 [N, C, H, W]
     * @param output 输出张量，形状 [N, C, 1, 1]（若 keepdim 为真，否则为 [N, C]）
     * @return TensorStatus
     * @note 输出空间维度被池化为1，因此无需指定窗口大小和步长。
     */
    TensorStatus tensor_global_avg_pool2d(const Tensor *input, Tensor *output);

    /**
     * @brief 二维全局最大池化
     */
    TensorStatus tensor_global_max_pool2d(const Tensor *input, Tensor *output);

    /* ==================== 归一化函数 ==================== */

    /**
     * @brief 批归一化（推理模式）
     * @param x       输入张量，形状一般为 [N, C, ...]
     * @param mean    均值，形状 [C]
     * @param var     方差，形状 [C]
     * @param scale   缩放因子，形状 [C]（可为 NULL，默认为1）
     * @param bias    偏置，形状 [C]（可为 NULL，默认为0）
     * @param epsilon 小常数，防止除零
     * @param y       输出张量，形状同 x
     * @return TensorStatus
     * @note 训练模式需要额外维护均值和方差的移动平均，本函数仅为推理使用。
     */
    TensorStatus tensor_batchnorm(const Tensor *x, const Tensor *mean, const Tensor *var,
                                  const Tensor *scale, const Tensor *bias,
                                  float epsilon, Tensor *y);

    /**
     * @brief 层归一化（对最后一维进行归一化）
     * @param x       输入张量
     * @param scale   缩放因子，形状应与 x 的最后一维相同（可为 NULL）
     * @param bias    偏置，形状应与 x 的最后一维相同（可为 NULL）
     * @param epsilon 小常数
     * @param y       输出张量，形状同 x
     * @return TensorStatus
     */
    TensorStatus tensor_layernorm(const Tensor *x, const Tensor *scale, const Tensor *bias,
                                  float epsilon, Tensor *y);

    /**
     * @brief 实例归一化（对每个样本每个通道单独归一化）
     * @param x       输入张量，形状一般为 [N, C, H, W] 或 [N, C, D, H, W]
     * @param scale   缩放因子，形状 [C]（可为 NULL）
     * @param bias    偏置，形状 [C]（可为 NULL）
     * @param epsilon 小常数
     * @param y       输出张量，形状同 x
     * @return TensorStatus
     */
    TensorStatus tensor_instancenorm(const Tensor *x, const Tensor *scale, const Tensor *bias,
                                     float epsilon, Tensor *y);

    /**
     * @brief 组归一化（将通道分组，对每组内进行归一化）
     * @param x          输入张量，形状 [N, C, ...]
     * @param scale      缩放因子，形状 [C]（可为 NULL）
     * @param bias       偏置，形状 [C]（可为 NULL）
     * @param num_groups 分组数，必须能整除 C
     * @param epsilon    小常数
     * @param y          输出张量，形状同 x
     * @return TensorStatus
     */
    TensorStatus tensor_groupnorm(const Tensor *x, const Tensor *scale, const Tensor *bias,
                                  int num_groups, float epsilon, Tensor *y);

    /**
     * @brief 本地响应归一化 (LRN)
     * @param x     输入张量，形状 [N, C, H, W]
     * @param size  邻域大小（必须为奇数）
     * @param alpha 缩放因子
     * @param beta  指数
     * @param bias  偏置
     * @param y     输出张量，形状同 x
     * @return TensorStatus
     */
    TensorStatus tensor_lrn(const Tensor *x, int size, float alpha, float beta, float bias, Tensor *y);

    /* ==================== 激活函数 ==================== */

    /**
     * @brief ReLU: out = max(0, x)
     */
    TensorStatus tensor_relu(const Tensor *x, Tensor *out);

    /**
     * @brief Leaky ReLU: out = x if x>0 else alpha*x
     */
    TensorStatus tensor_leaky_relu(const Tensor *x, float alpha, Tensor *out);

    /**
     * @brief ELU: out = x if x>0 else alpha*(exp(x)-1)
     */
    TensorStatus tensor_elu(const Tensor *x, float alpha, Tensor *out);

    /**
     * @brief SELU: 自归一化ELU，固定 scale 和 alpha 参数
     */
    TensorStatus tensor_selu(const Tensor *x, float alpha, float scale, Tensor *out);

    /**
     * @brief GELU (高斯误差线性单元，近似实现)
     */
    TensorStatus tensor_gelu(const Tensor *x, Tensor *out);

    /**
     * @brief Swish: out = x * sigmoid(x)
     */
    TensorStatus tensor_swish(const Tensor *x, Tensor *out);

    /**
     * @brief Mish: out = x * tanh(softplus(x))
     */
    TensorStatus tensor_mish(const Tensor *x, Tensor *out);

    /**
     * @brief Softplus: out = log(1 + exp(x))
     */
    TensorStatus tensor_softplus(const Tensor *x, Tensor *out);

    /**
     * @brief Softsign: out = x / (1 + |x|)
     */
    TensorStatus tensor_softsign(const Tensor *x, Tensor *out);

    /**
     * @brief Hardswish: out = x * clamp(x+3,0,6) / 6
     */
    TensorStatus tensor_hardswish(const Tensor *x, Tensor *out);

    /**
     * @brief Hardsigmoid: out = clamp(x+3,0,6) / 6
     */
    TensorStatus tensor_hardsigmoid(const Tensor *x, Tensor *out);

    /**
     * @brief PReLU (参数化ReLU): out = x if x>0 else alpha*x，其中 alpha 为可学习参数张量
     * @param x     输入张量
     * @param alpha 参数张量，通常形状与 x 相同或可广播
     * @param out   输出张量
     */
    TensorStatus tensor_prelu(const Tensor *x, const Tensor *alpha, Tensor *out);

    /* ==================== 其他层 ==================== */

    /**
     * @brief 全连接层: y = input @ weight^T + bias
     * @param input  输入张量，最后一维大小为 in_features
     * @param weight 权重张量，形状 [out_features, in_features]
     * @param bias   偏置张量，形状 [out_features]（可为 NULL）
     * @param output 输出张量，形状为 input 除最后一维外加上 [out_features]
     * @return TensorStatus
     */
    TensorStatus tensor_linear(const Tensor *input, const Tensor *weight, const Tensor *bias,
                               Tensor *output);

    /**
     * @brief Dropout（训练模式）
     * @param x        输入张量
     * @param p        丢弃概率 (0 <= p < 1)
     * @param training 非零表示训练模式（随机丢弃），零表示推理模式（直接返回 x）
     * @param out      输出张量，形状同 x
     * @return TensorStatus
     */
    TensorStatus tensor_dropout(const Tensor *x, float p, int training, Tensor *out);

    /**
     * @brief Softmax
     * @param x    输入张量
     * @param axis 应用 softmax 的轴
     * @param out  输出张量，形状同 x
     * @return TensorStatus
     */
    TensorStatus tensor_softmax(const Tensor *x, int axis, Tensor *out);

    /**
     * @brief LogSoftmax
     * @param x    输入张量
     * @param axis 轴
     * @param out  输出张量，形状同 x
     * @return TensorStatus
     */
    TensorStatus tensor_log_softmax(const Tensor *x, int axis, Tensor *out);

    /**
     * @brief 插值模式枚举
     */
    typedef enum
    {
        UPSAMPLE_NEAREST, //!< 最近邻插值
        UPSAMPLE_LINEAR,  //!< 线性插值（双线性/三线性）
        UPSAMPLE_CUBIC    //!< 三次插值
    } InterpMode;

    /**
     * @brief 2D 上采样
     * @param x            输入张量 [N, C, H, W]
     * @param scale_h      高度缩放因子（正整数）
     * @param scale_w      宽度缩放因子（正整数）
     * @param mode         插值模式
     * @param align_corners 是否对角线对齐（仅对 LINEAR/CUBIC 有效）
     * @param out          输出张量，形状 [N, C, H*scale_h, W*scale_w]
     * @return TensorStatus
     */
    TensorStatus tensor_upsample2d(const Tensor *x, int scale_h, int scale_w,
                                   InterpMode mode, int align_corners, Tensor *out);

    /**
     * @brief 2D 最大反池化（MaxUnpool2d）
     * @param x           输入张量 [N, C, H, W]
     * @param indices     最大池化时记录的索引张量，形状同 x，元素为 float 表示的整数索引（展平位置）
     * @param output_size 目标输出空间尺寸 [out_h, out_w]（可为 NULL，若为 NULL 则根据步长等自动计算）
     * @param out         输出张量 [N, C, out_h, out_w]
     * @return TensorStatus
     */
    TensorStatus tensor_max_unpool2d(const Tensor *x, const Tensor *indices, const int *output_size, Tensor *out);

    /**
     * @brief 2D 自适应平均池化（输出尺寸任意指定）
     * @param x           输入张量 [N, C, H, W]
     * @param output_size 目标输出尺寸 [out_h, out_w]
     * @param out         输出张量 [N, C, out_h, out_w]
     * @return TensorStatus
     */
    TensorStatus tensor_adaptive_avg_pool2d(const Tensor *x, const int *output_size, Tensor *out);

    /**
     * @brief 嵌入层（查找表）
     * @param input  输入索引张量，元素为 float 存储的整数，任意形状
     * @param weight 权重张量，形状 [vocab_size, embedding_dim]
     * @param padding_idx 可选的填充索引，该位置输出全0（-1 表示不使用）
     * @param out    输出张量，形状为 input.shape + [embedding_dim]
     * @return TensorStatus
     */
    TensorStatus tensor_embedding(const Tensor *input, const Tensor *weight,
                                  int padding_idx, Tensor *out);

    /* ---------- 上采样 ---------- */

    /**
     * @brief 1D 上采样
     * @param x            输入张量 [N, C, L]
     * @param scale        长度缩放因子（正整数）
     * @param mode         插值模式（最近邻、线性、三次）
     * @param align_corners 是否对齐角点（仅对 LINEAR/CUBIC 有效）
     * @param out          输出张量 [N, C, L*scale]
     * @return TensorStatus
     */
    TensorStatus tensor_upsample1d(const Tensor *x, int scale,
                                   InterpMode mode, int align_corners,
                                   Tensor *out);

    /**
     * @brief 3D 上采样
     * @param x            输入张量 [N, C, D, H, W]
     * @param scale_d      深度缩放因子
     * @param scale_h      高度缩放因子
     * @param scale_w      宽度缩放因子
     * @param mode         插值模式
     * @param align_corners 是否对齐角点
     * @param out          输出张量 [N, C, D*scale_d, H*scale_h, W*scale_w]
     * @return TensorStatus
     */
    TensorStatus tensor_upsample3d(const Tensor *x,
                                   int scale_d, int scale_h, int scale_w,
                                   InterpMode mode, int align_corners,
                                   Tensor *out);

    /* ---------- 自适应平均池化 ---------- */

    /**
     * @brief 1D 自适应平均池化
     * @param x           输入张量 [N, C, L]
     * @param output_size 目标长度（长度为1的数组）
     * @param out         输出张量 [N, C, output_size]
     * @return TensorStatus
     */
    TensorStatus tensor_adaptive_avg_pool1d(const Tensor *x,
                                            const int *output_size,
                                            Tensor *out);

    /**
     * @brief 3D 自适应平均池化
     * @param x           输入张量 [N, C, D, H, W]
     * @param output_size 目标尺寸 [out_d, out_h, out_w]（长度为3的数组）
     * @param out         输出张量 [N, C, out_d, out_h, out_w]
     * @return TensorStatus
     */
    TensorStatus tensor_adaptive_avg_pool3d(const Tensor *x,
                                            const int *output_size,
                                            Tensor *out);

    /* ---------- 最大反池化 ---------- */

    /**
     * @brief 1D 最大反池化
     * @param x           输入张量 [N, C, L]（池化后的值）
     * @param indices     索引张量，形状同 x，元素为展平位置（float 存储整数）
     * @param output_size 目标输出尺寸 [out_len]（长度为1的数组）
     * @param out         输出张量 [N, C, out_len]
     * @return TensorStatus
     */
    TensorStatus tensor_max_unpool1d(const Tensor *x, const Tensor *indices,
                                     const int *output_size, Tensor *out);

    /**
     * @brief 3D 最大反池化
     * @param x           输入张量 [N, C, D, H, W]
     * @param indices     索引张量，形状同 x
     * @param output_size 目标输出尺寸 [out_d, out_h, out_w]（长度为3的数组）
     * @param out         输出张量 [N, C, out_d, out_h, out_w]
     * @return TensorStatus
     */
    TensorStatus tensor_max_unpool3d(const Tensor *x, const Tensor *indices,
                                     const int *output_size, Tensor *out);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_NN_OPS_H