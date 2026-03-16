#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <stddef.h>
#include <stdint.h>

/**
 * @file tensor.h
 * @brief 核心张量数据结构与基础函数
 *
 * 本文件定义了张量的核心结构、状态码枚举以及基础的创建、销毁、视图等操作。
 * 所有函数均返回 TensorStatus 以指示执行状态。
 */

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief 张量操作状态码
     */
    typedef enum
    {
        TENSOR_OK = 0,                  // 操作成功
        TENSOR_ERR_MEMORY,              // 内存分配失败
        TENSOR_ERR_SHAPE_MISMATCH,      // 形状不兼容
        TENSOR_ERR_INVALID_PARAM,       // 参数无效
        TENSOR_ERR_UNSUPPORTED,         // 不支持的操作
        TENSOR_ERR_NULL_PTR,            // 空指针
        TENSOR_ERR_INDEX_OUT_OF_BOUNDS, // 索引越界
        TENSOR_ERR_DIV_BY_ZERO,         // 除零错误
        TENSOR_ERR_NOT_IMPLEMENTED      // 未实现
    } TensorStatus;

    /**
     * @brief 宏定义：M_PI
     * π值的定义，因为不是c++，math头文件中没有定义
     */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

    /**
     * @brief 宏定义：TENSOR_MAX_DIM
     * 在从头编译时有效，控制张量维度数量上限。若需要增加维度数量，请修改此宏定义并重新编译。
     */
#ifndef TENSOR_MAX_DIM
#define TENSOR_MAX_DIM 8
#endif

    /**
     * @brief 将状态码转换为可读字符串
     * @param status 状态码
     * @return 字符串描述
     */
    const char *tensor_status_to_string(TensorStatus status);

    /**
     * @brief 张量结构
     *
     * 张量的核心数据结构，支持引用计数、视图、写时拷贝等机制。
     * 所有字段均设计为公开，以便高级用户直接访问，但需谨慎操作。
     */
    typedef struct
    {
        float *data;           /* 数据指针，指向实际存储的 float 数组，按行主序排列。若 ref_count 非 NULL，则由库管理；若为 NULL，则由外部管理。*/
        int *dims;             /* 维度数组，长度为 ndim，表示每个维度的大小。由库分配，当 owns_dims_strides 为真时需释放。*/
        int *strides;          /* 步长数组，长度为 ndim，表示每个维度增加1时，在 data 中需跳过的元素个数。若为 NULL，表示张量是连续的（即按行主序排列）。*/
        int ndim;              /* 维度数量，即 dims 和 strides 的长度。可以为 0，表示标量张量（size=1）。*/
        size_t size;           /* 总元素个数，等于各维度大小的乘积。对于标量，size=1。*/
        int *ref_count;        /* 指向数据块的引用计数器。多个张量可共享同一数据块。当 ref_count 为 NULL 时，表示数据由外部管理，库不会自动释放 data。当 ref_count 非 NULL 时，*ref_count 表示共享该数据块的张量数量。*/
        int owns_dims_strides; /* 布尔值，指示当前张量是否拥有 dims 和 strides 数组。若为真，在 tensor_destroy 中会 free(dims) 和 free(strides)；若为假，则不会释放（例如视图创建时可能共享原张量的 dims/strides）。*/
    } Tensor;

    /**
     * @brief 创建一个新张量，所有元素初始化为0
     *
     * @param ndim 维度数量
     * @param dims 各维度大小数组，长度必须为 ndim
     * @return 成功返回张量指针，失败返回 NULL
     *
     * @note 张量的数据引用计数初始为1，调用者需在不再使用时调用 tensor_destroy。
     */
    Tensor *tensor_create(int ndim, const int *dims);

    /**
     * @brief 包装外部数据创建一个张量（不拷贝数据）
     *
     * @param data 外部数据指针，必须保证在张量使用期间有效
     * @param ndim 维度数量
     * @param dims 各维度大小数组
     * @param strides 各维度步长（以元素个数计），若为 NULL 则视为连续存储（行主序）
     * @return 成功返回张量指针，失败返回 NULL
     *
     * @note 此张量不管理数据内存，销毁时不会释放 data。
     */
    Tensor *tensor_wrap(float *data, int ndim, const int *dims, const int *strides);

    /**
     * @brief 从外部数组拷贝数据创建新张量
     *
     * @param data 源数据指针，长度至少为 total_size
     * @param ndim 维度数量
     * @param dims 各维度大小数组
     * @return 成功返回张量指针，失败返回 NULL
     */
    Tensor *tensor_from_array(const float *data, int ndim, const int *dims);

    /**
     * @brief 销毁张量，释放内存
     *
     * 若张量管理自己的数据（ref_count 非 NULL），则递减引用计数，计数归零时释放数据。
     * 同时释放 dims 和 strides 数组（如果 owns_dims_strides 为真）。
     *
     * @param t 张量指针
     */
    void tensor_destroy(Tensor *t);

    /**
     * @brief 释放张量内部资源，但不释放张量结构体本身
     *
     * 该函数用于清理栈上分配的 Tensor 结构体（或任何不由 tensor_create 分配的张量）。
     * 它会递减引用计数，并在计数为零时释放数据内存，同时释放 dims 和 strides 数组（如果 owns_dims_strides 为真）。
     * 与 tensor_destroy 不同，它不会 free(t) 本身，因此适用于 t 是栈变量或嵌入在其他结构中的情况。
     * 调用后，t 的字段将被置为 NULL，防止误用。
     *
     * @param t 目标张量指针
     */
    void tensor_cleanup(Tensor *t);

    /**
     * @brief 深拷贝张量，创建完全独立的新张量
     *
     * @param src 源张量
     * @return 成功返回新张量指针，失败返回 NULL
     */
    Tensor *tensor_clone(const Tensor *src);

    /**
     * @brief 创建源张量的视图（共享数据）
     *
     * @param src 源张量
     * @param ndim 视图的维度数量
     * @param dims 视图的维度数组
     * @param strides 视图的步长数组，若为 NULL 则假设视图连续（按 dims 自动计算）
     * @return 成功返回新张量指针（视图），失败返回 NULL
     *
     * @note 视图的数据与源张量共享，引用计数会增加。
     */
    Tensor *tensor_view(const Tensor *src, int ndim, const int *dims, const int *strides);

    /**
     * @brief 将数据从源张量复制到目标张量（要求形状相同）
     *
     * 若目标张量的引用计数为 1 且管理自己的数据，则直接覆盖；否则触发写时拷贝。
     *
     * @param dst 目标张量
     * @param src 源张量
     * @return TensorStatus
     */
    TensorStatus tensor_copy(Tensor *dst, const Tensor *src);

    /**
     * @brief 确保张量数据连续
     *
     * 若张量已连续，无操作；否则创建新的连续副本（可能改变引用关系）。
     *
     * @param t 张量
     * @return TensorStatus
     */
    TensorStatus tensor_contiguous(Tensor *t);

    /**
     * @brief 获取张量的维度数量
     * @param t 张量
     * @return 维度数
     */
    int tensor_ndim(const Tensor *t);

    /**
     * @brief 获取张量的维度数组指针
     * @param t 张量
     * @return 维度数组，长度为 tensor_ndim(t)
     */
    const int *tensor_dims(const Tensor *t);

    /**
     * @brief 获取张量的步长数组指针
     * @param t 张量
     * @return 步长数组，若为 NULL 表示张量连续
     */
    const int *tensor_strides(const Tensor *t);

    /**
     * @brief 获取张量的总元素个数
     * @param t 张量
     * @return 元素个数
     */
    size_t tensor_size(const Tensor *t);

    /**
     * @brief 获取指定轴的大小（支持负索引）
     * @param t 张量
     * @param axis 轴索引，-1 表示最后一维，以此类推
     * @return 该轴大小，若轴无效返回 -1
     */
    int tensor_dim_size(const Tensor *t, int axis);

    /**
     * @brief 根据索引计算元素在数据中的偏移量
     * @param t 张量
     * @param indices 索引数组，长度必须等于张量维度
     * @return 元素偏移量（以元素个数计），若索引越界返回 SIZE_MAX
     */
    size_t tensor_offset(const Tensor *t, const int *indices);

    /**
     * @brief 确保张量拥有独占的数据副本（写时拷贝）
     *
     * 若张量引用计数 >1，则复制数据并更新引用计数。
     * 若张量为外部数据（ref_count == NULL），则直接返回（假定可写）。
     *
     * @param t 目标张量
     * @return TensorStatus
     *
     * @note 此函数为内部实现，仅在测试或特殊场景下直接调用。
     *       常规修改操作（如 tensor_copy、tensor_set_item）会自动调用。
     */
    TensorStatus tensor_make_unique(Tensor *t);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_TENSOR_H