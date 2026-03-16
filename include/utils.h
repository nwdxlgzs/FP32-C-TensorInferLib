#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file utils.h
     * @brief 调试工具与内部通用辅助函数
     *
     * 本文件包含两类函数：
     * 1. 调试与实用工具（打印、保存、初始化等），供用户调用。
     * 2. 内部辅助函数和通用迭代器，用于减少库中各模块的代码重复。
     */

    /* ==================== 函数类型别名 ==================== */

    typedef float (*UnaryOp)(float);                 // 一元操作符：输入一个 float，返回一个 float
    typedef float (*BinaryOp)(float, float);         // 二元操作符：输入两个 float，返回一个 float
    typedef float (*TernaryOp)(float, float, float); // 三元操作符：输入三个 float，返回一个 float

    /* ==================== 调试与实用工具（原有） ==================== */

    /**
     * @brief 打印张量信息（形状、部分数据）
     * @param t 张量
     * @param name 打印时显示的名称（可为 NULL）
     * @param max_elements 最多打印的元素个数（-1 表示全部）
     */
    void tensor_print(const Tensor *t, const char *name, int max_elements);

    /**
     * @brief 将张量保存到二进制文件
     * @param t 张量
     * @param filename 文件名
     * @return TensorStatus
     */
    TensorStatus tensor_save(const Tensor *t, const char *filename);

    /**
     * @brief 从二进制文件加载张量
     * @param t 输出张量指针（需先置 NULL，函数内部分配）
     * @param filename 文件名
     * @return TensorStatus
     */
    TensorStatus tensor_load(Tensor **t, const char *filename);

    /**
     * @brief 比较两个张量是否近似相等
     * @param a 张量 a
     * @param b 张量 b
     * @param rtol 相对容忍误差
     * @param atol 绝对容忍误差
     * @return 1 表示近似相等，0 表示不相等
     */
    int tensor_allclose(const Tensor *a, const Tensor *b, float rtol, float atol);

    /**
     * @brief 检查张量是否包含 NaN
     * @param t 张量
     * @return 1 包含 NaN，0 不包含
     */
    int tensor_has_nan(const Tensor *t);

    /**
     * @brief 检查张量是否包含 Inf
     * @param t 张量
     * @return 1 包含 Inf，0 不包含
     */
    int tensor_has_inf(const Tensor *t);

    /**
     * @brief 用常数填充张量
     * @param t 张量
     * @param value 常数值
     * @return TensorStatus
     */
    TensorStatus tensor_fill(Tensor *t, float value);

    /**
     * @brief 用正态分布初始化张量（直接填充，不经过随机函数封装）
     * @param t 张量
     * @param mean 均值
     * @param std 标准差
     * @return TensorStatus
     */
    TensorStatus tensor_normal_init(Tensor *t, float mean, float std);

    /**
     * @brief 用均匀分布初始化张量
     * @param t 张量
     * @param low 下界
     * @param high 上界
     * @return TensorStatus
     */
    TensorStatus tensor_uniform_init(Tensor *t, float low, float high);

    /**
     * @brief Xavier 初始化（均匀分布，范围 sqrt(6/(fan_in+fan_out))）
     * @param t 张量
     * @param fan_in 输入通道数
     * @param fan_out 输出通道数
     * @return TensorStatus
     */
    TensorStatus tensor_xavier_init(Tensor *t, int fan_in, int fan_out);

    /**
     * @brief 计算多个形状的广播后形状
     * @param dims 形状数组指针（每个元素为 int*）
     * @param ndims 每个形状的维度数数组
     * @param num_tensors 张量个数
     * @param[out] out_dims 输出形状数组（长度至少为最大维度）
     * @param[out] out_ndim 输出维度数
     * @return 1 表示可广播，0 表示不可广播
     */
    int util_broadcast_shapes(const int *dims[], const int ndims[], int num_tensors,
                              int *out_dims, int *out_ndim);

    /**
     * @brief 检查两个形状是否可广播，并计算广播后的形状
     *
     * 广播规则：
     * - 从尾部维度开始对齐，较短的形状在前补1。
     * - 对应维度相等或其中一个为1时可广播。
     * - 输出维度取两者中的较大值。
     *
     * @param dims_a 形状a的维度数组
     * @param ndim_a 形状a的维度数
     * @param dims_b 形状b的维度数组
     * @param ndim_b 形状b的维度数
     * @param[out] out_dims 输出形状数组（长度至少为 max(ndim_a, ndim_b)）
     * @param[out] out_ndim 输出形状的维度数
     * @return 1 表示可广播，0 表示不可广播
     */
    int util_broadcast_shape(const int *dims_a, int ndim_a,
                             const int *dims_b, int ndim_b,
                             int *out_dims, int *out_ndim);

    /**
     * @brief 填充 padded 步长：前导维度补0，并将大小为1的维度步长置0
     *
     * 该函数用于广播操作，生成与广播后形状对齐的步长数组。
     * 若原张量步长为 NULL，则自动计算连续步长；否则使用已有步长。
     * 若某维度大小为1，步长强制设为0，以实现广播。
     *
     * @param t 源张量
     * @param out_ndim 广播后的维度数
     * @param out_dims 广播后的形状
     * @param[out] padded_strides 输出步长数组（长度至少为 out_ndim）
     */
    void util_fill_padded_strides(const Tensor *t, int out_ndim,
                                  const int *out_dims, int *padded_strides);

    /**
     * @brief 获取张量的有效步长（若 strides==NULL 则计算连续步长）
     *
     * 如果张量 strides 字段非 NULL，直接复制；否则按行主序计算连续步长。
     * 连续步长计算公式：最后一个维度步长为1，向前依次乘以维度大小。
     *
     * @param t 张量
     * @param[out] strides_out 输出步长数组（长度至少为 t->ndim）
     */
    void util_get_effective_strides(const Tensor *t, int *strides_out);

    /**
     * @brief 归一化轴索引（支持负索引）
     *
     * 负索引从末尾开始计数，例如 -1 表示最后一维。
     *
     * @param axis 原始轴索引
     * @param ndim 张量的维度数
     * @return 归一化后的非负轴索引（0 ~ ndim-1），若无效返回 -1
     */
    int util_normalize_axis(int axis, int ndim);

    /**
     * @brief 根据坐标和步长计算偏移量（元素个数）
     *
     * @param coords 坐标数组（长度等于 ndim）
     * @param strides 步长数组（长度等于 ndim）
     * @param ndim 维度数
     * @return 数据偏移量，可用于索引 data 数组
     */
    size_t util_offset_from_coords(const int *coords, const int *strides, int ndim);

    /* ==================== 通用迭代器 ==================== */

    /**
     * @brief 一元运算通用迭代器
     *
     * 对输入张量 x 的每个元素应用操作 op，结果写入输出张量 out。
     * 要求 x 与 out 形状完全相同。
     * 内部会自动调用 tensor_make_unique 确保 out 独占数据。
     *
     * @param x 输入张量
     * @param out 输出张量
     * @param op 一元操作函数指针
     * @return TensorStatus 状态码
     */
    TensorStatus util_unary_op_general(const Tensor *x, Tensor *out, UnaryOp op);

    /**
     * @brief 二元运算通用迭代器（支持广播）
     *
     * 根据广播规则对 a 和 b 进行逐元素操作，结果写入 out。
     * 广播形状由 a 和 b 的形状决定，输出 out 的形状必须与广播结果一致。
     * 内部会自动调用 tensor_make_unique 确保 out 独占数据。
     *
     * @param a 输入张量 a
     * @param b 输入张量 b
     * @param out 输出张量
     * @param op 二元操作函数指针
     * @return TensorStatus 状态码
     */
    TensorStatus util_binary_op_general(const Tensor *a, const Tensor *b,
                                        Tensor *out, BinaryOp op);

    /**
     * @brief 三元运算通用迭代器（支持广播）
     *
     * 根据广播规则对 a、b、c 进行逐元素操作，结果写入 out。
     * 广播形状由三个张量的形状决定，输出 out 的形状必须与广播结果一致。
     * 内部会自动调用 tensor_make_unique 确保 out 独占数据。
     *
     * @param a 输入张量 a
     * @param b 输入张量 b
     * @param c 输入张量 c
     * @param out 输出张量
     * @param op 三元操作函数指针
     * @return TensorStatus 状态码
     */
    TensorStatus util_ternary_op_general(const Tensor *a, const Tensor *b,
                                         const Tensor *c, Tensor *out, TernaryOp op);

    /**
     * @brief 将标量包装为零维张量后调用二元通用迭代器
     *
     * 构造一个临时的零维张量（标量）包装标量值，然后调用 util_binary_op_general。
     * 该函数不会复制数据，调用者需确保 scalar 在调用期间有效。
     *
     * @param a 输入张量
     * @param scalar 标量值
     * @param out 输出张量
     * @param op 二元操作函数指针 (BinaryOp)
     * @return TensorStatus
     */
    TensorStatus util_binary_op_scalar(const Tensor *a, float scalar,
                                       Tensor *out, BinaryOp op);

    /**
     * @brief 对张量的每个元素调用生成函数（无输入）
     * @param t 输出张量
     * @param gen 生成函数，接受用户数据，返回 float
     * @param user_data 用户数据指针
     */
    TensorStatus util_generate_op(Tensor *t, float (*gen)(void *), void *user_data);

    /**
     * @brief 检查张量是否连续（行主序）
     * @param t 张量
     * @return 1 表示连续，0 表示不连续或空指针
     */
    int util_is_contiguous(const Tensor *t);

    /**
     * @brief 复制整数数组（分配新内存）
     * @param src 源数组
     * @param n 元素个数
     * @return 新分配的数组副本，失败返回 NULL
     */
    int *util_copy_ints(const int *src, int n);

    /**
     * @brief 计算连续存储的步长（行主序）
     * @param dims 维度数组
     * @param ndim 维度数
     * @return 新分配的步长数组，失败返回 NULL
     */
    int *util_calc_contiguous_strides(const int *dims, int ndim);

    /**
     * @brief 计算张量的总元素数
     * @param dims 维度数组
     * @param ndim 维度数
     * @return 总元素数
     */
    size_t util_calc_size(const int *dims, int ndim);

    /**
     * @brief 检查两个形状是否完全相同
     * @param a 形状 a 的维度数组
     * @param b 形状 b 的维度数组
     * @param ndim 维度数
     * @return 1 表示相等，0 表示不等
     */
    int util_shapes_equal(const int *a, const int *b, int ndim);

    /**
     * @brief 按行主序递增坐标，并返回是否已结束
     * @param coords 坐标数组（会被修改）
     * @param dims 各维度大小
     * @param ndim 维度数
     * @return 0 表示仍有效（未结束），1 表示已遍历完所有坐标
     */
    int util_increment_coords(int *coords, const int *dims, int ndim);

    /**
     * @brief 归一化轴索引（支持负索引）
     *
     * 将用户传入的轴索引转换为非负索引。负索引从末尾开始计数，
     * 例如 -1 表示最后一维。若转换后的索引超出有效范围，返回 -1。
     *
     * @param axis 原始轴索引（可为负）
     * @param ndim 张量的维度数
     * @return 归一化后的非负轴索引（0 ~ ndim-1），若无效返回 -1
     */
    int util_normalize_axis(int axis, int ndim);

    /**
     * @brief 将行主序线性索引转换为多维坐标（假设连续存储）
     * @param linear 线性索引
     * @param dims 维度数组
     * @param ndim 维度数
     * @param[out] coords 输出的坐标数组（长度 ndim）
     */
    void util_coords_from_linear(size_t linear, const int *dims, int ndim, int *coords);

    /**
     * @brief 检查两个张量是否指向同一数据内存
     * @return 1 是，0 否
     */
    int util_same_data(const Tensor *a, const Tensor *b);

    /**
     * @brief 将张量所有元素设置为 0（按步长遍历，支持非连续张量）
     * @param t 目标张量（必须已通过 tensor_make_unique 确保独占）
     */
    void util_clear_tensor(Tensor *t);

    /**
     * @brief 按逻辑顺序打印张量（支持非连续视图）
     * @param t 张量
     * @param name 名称（可为 NULL）
     * @param max_elements 最多打印元素个数，-1 表示全部
     */
    void tensor_print_logical(const Tensor *t, const char *name, int max_elements);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_UTILS_H