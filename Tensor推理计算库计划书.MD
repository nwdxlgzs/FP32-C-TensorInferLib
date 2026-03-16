纯C浮点张量推理库完整设计文档

1. 项目概述

本项目是一个纯C语言实现的、面向CPU的浮点（float32）张量运算库，用于深度学习模型推理。库提供完整的底层张量操作，包括内存管理、基础算术、神经网络层、形状变换、索引、随机数生成等。

核心设计原则：

· 标准化朴素实现：所有运算均优先采用最直观的循环实现，不引入任何SIMD指令或平台特定的优化。代码简洁、可读性强，保证跨平台一致性和可移植性。
· 高质量数据复用：通过引用计数和视图机制，最大限度减少不必要的数据拷贝。
· 良好的内存管理：自动管理内存生命周期，避免泄漏和悬挂指针，支持写时拷贝（Copy-on-Write）以安全共享数据。

2. 核心数据结构与基础函数

2.1 状态码定义

```c
// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TENSOR_OK = 0,
    TENSOR_ERR_MEMORY,           // 内存分配失败
    TENSOR_ERR_SHAPE_MISMATCH,   // 形状不兼容
    TENSOR_ERR_INVALID_PARAM,    // 参数无效
    TENSOR_ERR_UNSUPPORTED,      // 不支持的操作
    TENSOR_ERR_NULL_PTR,         // 空指针
    TENSOR_ERR_INDEX_OUT_OF_BOUNDS, // 索引越界
    TENSOR_ERR_DIV_BY_ZERO,      // 除零错误
    TENSOR_ERR_NOT_IMPLEMENTED   // 未实现
} TensorStatus;

const char* tensor_status_to_string(TensorStatus status);
```

2.2 张量结构

```c
typedef struct  {
    float* data;           // 数据指针（行主序）
    int* dims;             // 各维度大小
    int* strides;          // 各维度步长（以元素个数计），若为NULL则视为连续存储
    int ndim;              // 维度数量
    size_t size;           // 总元素个数
    int* ref_count;        // 引用计数器（指向数据块的引用计数，为NULL表示数据由外部管理）
    int owns_dims_strides; // 是否拥有dims和strides数组
} Tensor;
```

2.3 基础函数

函数 描述
Tensor* tensor_create(int ndim, const int* dims) 创建新张量，数据初始化为0，ref_count新分配并置1
Tensor* tensor_wrap(float* data, int ndim, const int* dims, const int* strides) 包装外部数据，不拷贝，ref_count为NULL，数据由外部管理
Tensor* tensor_from_array(float* data, int ndim, const int* dims) 从外部数组拷贝数据创建张量，ref_count新分配并置1
void tensor_destroy(Tensor* t) 销毁张量：若ref_count非NULL，递减并检查是否释放数据；释放dims和strides（如果owns_dims_strides）
Tensor* tensor_clone(const Tensor* src) 深拷贝张量：分配新数据并复制，ref_count新分配并置1
Tensor* tensor_view(const Tensor* src, int ndim, const int* dims, const int* strides) 创建视图（共享数据），增加src的引用计数，新张量的dims和strides为传入值（需分配内存）
TensorStatus tensor_copy(Tensor* dst, const Tensor* src) 复制数据（要求形状相同），若dst的ref_count为1且owns_data，则直接覆盖；否则触发写时拷贝
TensorStatus tensor_contiguous(Tensor* t) 确保张量数据连续：若已连续则无操作；否则创建新连续副本（可能改变引用关系）
int tensor_ndim(const Tensor* t) 获取维度数
const int* tensor_dims(const Tensor* t) 获取维度数组指针
const int* tensor_strides(const Tensor* t) 获取步长数组指针（可能NULL）
size_t tensor_size(const Tensor* t) 获取总元素个数
int tensor_dim_size(const Tensor* t, int axis) 获取指定轴的大小（支持负索引）
size_t tensor_offset(const Tensor* t, const int* indices) 根据索引计算元素偏移

写时拷贝（COW）策略：
任何可能修改张量数据的操作（如逐元素运算输出到已有张量、tensor_set_item、tensor_copy等）在执行前检查目标张量的ref_count：

· 若ref_count为NULL（外部数据），直接修改（需确保数据可写）。
· 若ref_count为1，直接修改（独占数据）。
· 若ref_count>1，则触发写时拷贝：分配新数据块，复制原数据，递减原引用计数，将目标张量的数据指针指向新块，新引用计数置1，然后执行修改。

3. 数学运算模块（朴素循环实现）

所有数学运算均采用最直观的嵌套循环实现，遍历张量元素，按行主序处理。广播通过步长处理或扩展为相同形状的循环实现。

3.1 逐元素一元运算

所有函数原型：TensorStatus tensor_op(const Tensor* x, Tensor* out);
out必须与x形状相同，或通过_alloc版本自动创建。若out与x共享数据且引用计数>1，自动进行写时拷贝。

函数 数学公式
tensor_neg out = -x
tensor_abs out =
tensor_sqrt out = sqrt(x)
tensor_rsqrt out = 1/sqrt(x)
tensor_exp out = exp(x)
tensor_log out = log(x)
tensor_log1p out = log(1+x)
tensor_sin out = sin(x)
tensor_cos out = cos(x)
tensor_tan out = tan(x)
tensor_asin out = asin(x)
tensor_acos out = acos(x)
tensor_atan out = atan(x)
tensor_sinh out = sinh(x)
tensor_cosh out = cosh(x)
tensor_tanh out = tanh(x)
tensor_asinh out = asinh(x)
tensor_acosh out = acosh(x)
tensor_atanh out = atanh(x)
tensor_ceil out = ceil(x)
tensor_floor out = floor(x)
tensor_round out = round(x)
tensor_trunc out = trunc(x)
tensor_sign out = sign(x)（-1,0,1）
tensor_square out = x^2
tensor_cube out = x^3
tensor_reciprocal out = 1/x
tensor_erf out = erf(x)
tensor_erfc out = erfc(x)

3.2 二元运算（支持广播）

原型：TensorStatus tensor_op(const Tensor* a, const Tensor* b, Tensor* out);
输出张量形状必须与广播结果一致。广播实现方式：通过计算每个输出元素对应的a和b的索引（考虑步长），使用循环遍历所有输出坐标。

函数 数学公式
tensor_add out = a + b
tensor_sub out = a - b
tensor_mul out = a * b
tensor_div out = a / b
tensor_pow out = a^b
tensor_maximum out = max(a, b)
tensor_minimum out = min(a, b)
tensor_fmod out = a % b
tensor_hypot out = sqrt(a^2 + b^2)
tensor_atan2 out = atan2(a, b)
tensor_copysign out =
tensor_nextafter out = nextafter(a, b)

3.3 标量版本

原型：TensorStatus tensor_op_scalar(const Tensor* a, float scalar, Tensor* out);
列表：

函数 描述
tensor_add_scalar out = a + scalar
tensor_sub_scalar out = a - scalar
tensor_mul_scalar out = a * scalar
tensor_div_scalar out = a / scalar
tensor_pow_scalar out = a^scalar
tensor_maximum_scalar out = max(a, scalar)
tensor_minimum_scalar out = min(a, scalar)

3.4 三元运算

函数 描述
tensor_clamp out = min(max(x, min), max)，min和max为张量，支持广播
tensor_clamp_scalar out = min(max(x, min_val), max_val)
tensor_where out = condition ? x : y，condition为布尔张量（0或1）

4. 归约操作

原型：TensorStatus tensor_op(const Tensor* x, Tensor* out, int axis, int keepdims);
axis=-1表示所有轴；keepdims非0则保持维度。归约通过遍历指定轴累加/比较实现。

函数 描述
tensor_sum 求和
tensor_mean 平均值
tensor_prod 乘积
tensor_max 最大值
tensor_min 最小值
tensor_argmax 最大值索引
tensor_argmin 最小值索引
tensor_any 逻辑或（非零视为True）
tensor_all 逻辑与
tensor_var 方差，unbiased参数控制是否使用样本方差（除n-1）
tensor_std 标准差，unbiased参数同方差
tensor_norm p范数，p为范数阶数（p=0表示非零个数）

5. 比较与逻辑运算

比较运算返回布尔张量（0.0或1.0）。支持广播，实现同二元运算。

函数 描述
tensor_equal a == b
tensor_not_equal a != b
tensor_less a < b
tensor_less_equal a <= b
tensor_greater a > b
tensor_greater_equal a >= b
tensor_equal_scalar a == scalar
tensor_not_equal_scalar a != scalar
tensor_less_scalar a < scalar
tensor_less_equal_scalar a <= scalar
tensor_greater_scalar a > scalar
tensor_greater_equal_scalar a >= scalar
tensor_logical_and a && b（a,b应为布尔）
tensor_logical_or a
tensor_logical_xor a ^ b
tensor_logical_not !a

6. 线性代数

函数 描述
tensor_matmul 矩阵乘法：a [..., M, K] × b [..., K, N] → [..., M, N]，支持批量广播
tensor_bmm 批量矩阵乘法，要求a和b均为3维[batch, M, K]和[batch, K, N]
tensor_dot 向量点积（两个1维张量）
tensor_outer 向量外积（1维张量）
tensor_tensordot 张量缩并：沿指定轴对求和，axes_a和axes_b为轴索引数组，naxes为轴对数
tensor_transpose 矩阵转置（交换最后两维）
tensor_permute 一般转置：按axes顺序重新排列轴
tensor_diag 提取对角线（输入2维）或创建对角阵（输入1维）
tensor_trace 迹，沿axis1和axis2的对角线和
tensor_inv 矩阵求逆（仅支持方阵，朴素高斯消元实现）

7. 神经网络层

所有层均采用朴素实现，例如卷积使用6层嵌套循环（批、输出通道、输出高、输出宽、输入通道、核高、核宽），步长和填充通过索引计算处理。

7.1 卷积参数

```c
typedef struct {
    int pad[3];            // 每个维度的填充（前后各一个数，如pad_h, pad_w, pad_d）
    int stride[3];         // 步长
    int dilation[3];       // 扩张率
    int groups;            // 分组数
} ConvParams;
```

卷积函数：

· tensor_conv1d(const Tensor* input, const Tensor* weight, const Tensor* bias, ConvParams params, Tensor* output)
· tensor_conv2d(const Tensor* input, const Tensor* weight, const Tensor* bias, ConvParams params, Tensor* output)
· tensor_conv3d(const Tensor* input, const Tensor* weight, const Tensor* bias, ConvParams params, Tensor* output)
· tensor_conv_transpose1d(const Tensor* input, const Tensor* weight, const Tensor* bias, ConvParams params, Tensor* output)
· tensor_conv_transpose2d(const Tensor* input, const Tensor* weight, const Tensor* bias, ConvParams params, Tensor* output)
· tensor_conv_transpose3d(const Tensor* input, const Tensor* weight, const Tensor* bias, ConvParams params, Tensor* output)

7.2 池化参数

```c
typedef enum {
    POOL_MAX,
    POOL_AVG,
    POOL_L2
} PoolType;

typedef struct {
    int kernel[3];       // 核大小
    int pad[3];          // 填充
    int stride[3];       // 步长
    int ceil_mode;       // 使用ceil计算输出尺寸
    int count_include_pad; // 平均池化是否包含填充
} PoolParams;
```

池化函数：

· tensor_pool1d(const Tensor* input, PoolType type, PoolParams params, Tensor* output)
· tensor_pool2d(const Tensor* input, PoolType type, PoolParams params, Tensor* output)
· tensor_pool3d(const Tensor* input, PoolType type, PoolParams params, Tensor* output)
· tensor_global_avg_pool2d(const Tensor* input, Tensor* output)
· tensor_global_max_pool2d(const Tensor* input, Tensor* output)

7.3 归一化

函数 描述
tensor_batchnorm(const Tensor* x, const Tensor* mean, const Tensor* var, const Tensor* scale, const Tensor* bias, float epsilon, Tensor* y) 批归一化（推理模式）
tensor_layernorm(const Tensor* x, const Tensor* scale, const Tensor* bias, float epsilon, Tensor* y) 层归一化（对最后一维）
tensor_instancenorm(const Tensor* x, const Tensor* scale, const Tensor* bias, float epsilon, Tensor* y) 实例归一化（对每个样本每个通道）
tensor_groupnorm(const Tensor* x, const Tensor* scale, const Tensor* bias, int num_groups, float epsilon, Tensor* y) 组归一化
tensor_lrn(const Tensor* x, int size, float alpha, float beta, float bias, Tensor* y) 本地响应归一化

7.4 激活函数

函数 公式
tensor_relu(const Tensor* x, Tensor* out) max(0, x)
tensor_leaky_relu(const Tensor* x, float alpha, Tensor* out) x>0 ? x : alpha*x
tensor_elu(const Tensor* x, float alpha, Tensor* out) x>0 ? x : alpha*(exp(x)-1)
tensor_selu(const Tensor* x, float alpha, float scale, Tensor* out) scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
tensor_gelu(const Tensor* x, Tensor* out) x * Φ(x)（近似）
tensor_swish(const Tensor* x, Tensor* out) x * sigmoid(x)
tensor_mish(const Tensor* x, Tensor* out) x * tanh(softplus(x))
tensor_softplus(const Tensor* x, Tensor* out) log(1+exp(x))
tensor_softsign(const Tensor* x, Tensor* out) x / (1+
tensor_hardswish(const Tensor* x, Tensor* out) x * clamp(x+3,0,6)/6
tensor_hardsigmoid(const Tensor* x, Tensor* out) clamp(x+3,0,6)/6
tensor_prelu(const Tensor* x, const Tensor* alpha, Tensor* out) max(0,x) + alpha * min(0,x)

7.5 其他层

函数 描述
tensor_linear(const Tensor* input, const Tensor* weight, const Tensor* bias, Tensor* output) 全连接：y = input @ weight^T + bias
tensor_dropout(const Tensor* x, float p, int training, Tensor* out) 训练时以概率p置零并缩放1/(1-p)，推理时直接返回x
tensor_softmax(const Tensor* x, int axis, Tensor* out) softmax
tensor_log_softmax(const Tensor* x, int axis, Tensor* out) log(softmax)
tensor_upsample2d(const Tensor* x, int scale_h, int scale_w, InterpMode mode, Tensor* out) 2D上采样，mode为UPSAMPLE_NEAREST, UPSAMPLE_LINEAR, UPSAMPLE_CUBIC

8. 形状操作

所有形状操作返回视图（共享数据）或新张量，遵循引用计数规则。视图通过调整步长实现，不复制数据。

函数 描述
tensor_reshape(Tensor* t, int ndim, const int* dims) 原地改变形状（要求元素总数不变，数据连续时才能成功）
tensor_reshape_view(const Tensor* src, int ndim, const int* dims, Tensor* dst) 返回视图（共享数据），增加引用计数
tensor_flatten(const Tensor* src, int start_axis, int end_axis, Tensor* dst) 将[start_axis, end_axis]展平，返回视图
tensor_squeeze(const Tensor* src, const int* axes, int num_axes, Tensor* dst) 移除指定轴（大小为1），axes为NULL则移除所有，返回视图
tensor_unsqueeze(const Tensor* src, int axis, Tensor* dst) 在axis处插入大小为1的新轴，返回视图
tensor_concat(const Tensor** inputs, int num_inputs, int axis, Tensor* output) 沿axis拼接多个张量，返回新张量
tensor_stack(const Tensor** inputs, int num_inputs, int axis, Tensor* output) 沿新轴堆叠多个张量，返回新张量
tensor_split(const Tensor* src, int axis, const int* sizes, int num_splits, Tensor** outputs) 沿axis分割成多个子张量，返回视图数组
tensor_slice(const Tensor* src, const int* starts, const int* ends, const int* steps, Tensor* dst) 切片（支持负索引、步长），返回视图
tensor_repeat(const Tensor* src, int axis, int repeats, Tensor* dst) 沿axis重复元素repeats次，返回新张量
tensor_tile(const Tensor* src, const int* reps, Tensor* dst) 按reps平铺整个张量，返回新张量
tensor_transpose_axes(const Tensor* src, const int* axes, Tensor* dst) 按axes顺序重新排列轴，返回视图
tensor_swapaxes(Tensor* t, int axis1, int axis2) 原地交换两个轴（需重新计算步长）
tensor_flip(const Tensor* src, const int* axes, int num_axes, Tensor* dst) 沿指定轴反转，返回视图
tensor_pad(const Tensor* src, const int* pad_widths, PadMode mode, float constant_value, Tensor* dst) 填充，mode为PAD_CONSTANT, PAD_REFLECT, PAD_REPLICATE, PAD_CIRCULAR，返回新张量
tensor_cumsum(const Tensor* src, int axis, Tensor* dst) 沿axis累积和，返回新张量
tensor_cumprod(const Tensor* src, int axis, Tensor* dst) 沿axis累积积，返回新张量

9. 索引与高级索引

函数 描述
tensor_get_item(const Tensor* src, const int* indices, float* out_value) 获取单个元素值
tensor_set_item(Tensor* dst, const int* indices, float value) 设置单个元素值（可能触发写时拷贝）
tensor_get_slice(const Tensor* src, const int* starts, const int* ends, const int* steps, Tensor* out) 切片获取子张量（返回视图）
tensor_advanced_index(const Tensor* src, const Tensor** indices, int num_indices, Tensor* out) 整数数组索引，每个索引张量形状需兼容，返回新张量
tensor_masked_select(const Tensor* src, const Tensor* mask, Tensor* out) 布尔掩码选择，返回1维新张量
tensor_index_put(Tensor* dst, const Tensor** indices, int num_indices, const Tensor* values) 按索引赋值（可能触发写时拷贝），values可广播
tensor_gather(const Tensor* src, int axis, const Tensor* index, Tensor* out) 沿axis根据index收集元素，返回新张量
tensor_scatter(Tensor* dst, int axis, const Tensor* index, const Tensor* src) 沿axis根据index将src的值放入dst（可能触发写时拷贝）

10. 随机数生成

使用标准C库的rand()实现，所有分布通过变换得到。

函数 描述
void tensor_random_seed(unsigned int seed) 设置全局随机种子
TensorStatus tensor_random_uniform(Tensor* t, float low, float high) 填充均匀分布[low, high)
TensorStatus tensor_random_normal(Tensor* t, float mean, float std) 填充正态分布（Box-Muller变换）
TensorStatus tensor_random_truncated_normal(Tensor* t, float mean, float std, float a, float b) 截断正态分布（拒绝采样）
TensorStatus tensor_random_bernoulli(Tensor* t, float p) 伯努利分布（生成0或1）
TensorStatus tensor_random_randint(Tensor* t, int low, int high) 填充随机整数[low, high)
TensorStatus tensor_shuffle(const Tensor* src, Tensor* dst) 沿第一维打乱，返回新张量（Fisher-Yates算法）

11. 调试与实用工具

函数 描述
void tensor_print(const Tensor* t, const char* name, int max_elements) 打印张量信息（形状、部分数据）
TensorStatus tensor_save(const Tensor* t, const char* filename) 保存张量到二进制文件
TensorStatus tensor_load(Tensor** t, const char* filename) 从二进制文件加载张量
int tensor_allclose(const Tensor* a, const Tensor* b, float rtol, float atol) 比较两个张量是否近似相等
int tensor_has_nan(const Tensor* t) 检查是否包含NaN
int tensor_has_inf(const Tensor* t) 检查是否包含Inf
TensorStatus tensor_fill(Tensor* t, float value) 用常数填充张量
TensorStatus tensor_normal_init(Tensor* t, float mean, float std) 用正态分布初始化
TensorStatus tensor_uniform_init(Tensor* t, float low, float high) 用均匀分布初始化
TensorStatus tensor_xavier_init(Tensor* t, int fan_in, int fan_out) Xavier初始化

12. 内存管理详解

12.1 引用计数机制

· 每个张量数据块关联一个引用计数器（ref_count），计数器在首次分配数据时创建并置1。
· 当通过tensor_view创建视图时，新张量的data指向原数据块，ref_count指向同一个计数器，并原子递增。
· tensor_clone创建完全独立的数据副本，计数器重新分配。
· tensor_destroy递减计数器，当计数器降至0时释放数据块。
· 外部数据包装（tensor_wrap）的ref_count为NULL，表示数据不由库管理，销毁时仅释放张头结构。

12.2 写时拷贝（Copy-on-Write）

· 任何可能修改张量数据的操作在写入前检查目标张量的引用计数，若>1则执行数据复制。
· 这确保了共享数据的安全，同时避免不必要的拷贝。

12.3 维度与步长的管理

· 每个张量拥有自己的dims和strides数组，由owns_dims_strides标志指示是否需要释放。
· 视图操作会为新张量分配新的dims和strides。

12.4 内存分配策略

· 使用标准malloc/free。
· 所有分配失败返回TENSOR_ERR_MEMORY。
· 提供_alloc系列函数自动创建输出张量，简化调用。

13. 文件组织

```
include/
  tensor.h          // 核心数据结构、基础函数、状态码
  math_ops.h        // 一元、二元、三元运算
  reduce_ops.h      // 归约运算
  compare_ops.h     // 比较逻辑
  linalg_ops.h      // 线性代数
  nn_ops.h          // 神经网络层
  shape_ops.h       // 形状操作
  indexing.h        // 索引
  random_ops.h      // 随机数
  utils.h           // 调试工具
src/
  tensor.c
  math_ops.c
  reduce_ops.c
  compare_ops.c
  linalg_ops.c
  nn_ops.c
  shape_ops.c
  indexing.c
  random_ops.c
  utils.c
tests/
  test_tensor.c
  test_math_ops.c
  test_reduce_ops.c
  test_compare_ops.c
  test_linalg_ops.c
  test_nn_ops.c
  test_shape_ops.c
  test_indexing.c
  test_random_ops.c
  test_utils.c
  runbuild_test_tensor.bat
  runbuild_test_math_ops.bat
  ...
```

14. 错误处理

所有API返回TensorStatus，调用者应检查返回值。使用tensor_status_to_string获取错误描述。

15. 灵活调整

项目会根据需求进行扩展，计划书只是纲领，实际以具体的代码文件为准。调试程序会携带宏-DDEBUG_TENSOR=1，会额外激活一些设置，只有编写test程序才可以使用他们，其他情况不要碰。

