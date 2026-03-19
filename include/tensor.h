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

    /* ==================== 状态码枚举 ==================== */

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
        TENSOR_ERR_SINGULAR_MATRIX,     // 奇异矩阵
        TENSOR_ERR_NOT_IMPLEMENTED      // 未实现
    } TensorStatus;

    /**
     * @brief 将状态码转换为可读字符串
     * @param status 状态码
     * @return 字符串描述
     */
    const char *tensor_status_to_string(TensorStatus status);

    /* ==================== 数学常量 ==================== */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

    /* ==================== 维度上限 ==================== */

#ifndef TENSOR_MAX_DIM
#define TENSOR_MAX_DIM 8
#endif

    /* ==================== 核心数据结构 ==================== */

    /**
     * @brief 张量结构
     *
     * 支持引用计数、视图、写时拷贝等机制。所有字段公开，便于高级用户直接访问。
     */
    typedef struct
    {
        float *data;           /* 数据指针，按行主序排列。若 ref_count 非 NULL，则由库管理；否则由外部管理。*/
        int *dims;             /* 维度数组，长度为 ndim。由库分配，当 owns_dims_strides 为真时需释放。*/
        int *strides;          /* 步长数组，长度为 ndim。若为 NULL，表示张量连续（行主序）。*/
        int ndim;              /* 维度数量（可为0，表示标量）。*/
        size_t size;           /* 总元素个数，等于各维度大小的乘积。*/
        int *ref_count;        /* 引用计数器。若为 NULL，数据由外部管理，库不会自动释放。*/
        int owns_dims_strides; /* 布尔值，指示当前张量是否拥有 dims 和 strides 数组。*/
    } Tensor;

    /* ==================== 创建与销毁 ==================== */

    /**
     * @brief 创建一个新张量，所有元素初始化为0
     * @param ndim 维度数量
     * @param dims 各维度大小数组
     * @return 成功返回张量指针，失败返回 NULL
     * @note 引用计数初始为1，需调用 tensor_destroy 释放。
     */
    Tensor *tensor_create(int ndim, const int *dims);

    /**
     * @brief 包装外部数据创建一个张量（不拷贝数据）
     * @param data 外部数据指针，必须保证在张量使用期间有效且可写（如需修改）
     * @param ndim 维度数量
     * @param dims 各维度大小数组
     * @param strides 各维度步长（以元素个数计），若为 NULL 则视为连续存储
     * @return 成功返回张量指针，失败返回 NULL
     * @note 此张量不管理数据内存，销毁时不会释放 data。用户需确保外部数据可写。
     */
    Tensor *tensor_wrap(float *data, int ndim, const int *dims, const int *strides);

    /**
     * @brief 从外部数组拷贝数据创建新张量
     * @param data 源数据指针，长度至少为 total_size
     * @param ndim 维度数量
     * @param dims 各维度大小数组
     * @return 成功返回张量指针，失败返回 NULL
     */
    Tensor *tensor_from_array(const float *data, int ndim, const int *dims);

    /**
     * @brief 销毁张量，释放内存
     * @param t 张量指针
     * @note 若引用计数归零，释放数据；若 owns_dims_strides 为真，释放 dims/strides。
     */
    void tensor_destroy(Tensor *t);

    /**
     * @brief 释放张量内部资源，但不释放张量结构体本身
     * @param t 目标张量指针（栈上分配或嵌入结构时使用）
     * @note 递减引用计数，计数为零时释放数据内存，并释放 dims/strides（若拥有）。
     *       调用后 t 的字段将被置为 NULL。
     */
    void tensor_cleanup(Tensor *t);

    /**
     * @brief 深拷贝张量，创建完全独立的新张量
     * @param src 源张量
     * @return 成功返回新张量指针，失败返回 NULL
     */
    Tensor *tensor_clone(const Tensor *src);

    /* ==================== 属性查询 ==================== */

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

    /* ==================== 视图与变形 ==================== */

    /**
     * @brief 创建源张量的视图（共享数据）
     * @param src 源张量
     * @param ndim 视图的维度数量
     * @param dims 视图的维度数组
     * @param strides 视图的步长数组，若为 NULL 则自动计算连续步长
     * @return 成功返回新张量指针（视图），失败返回 NULL
     * @note 视图数据与源张量共享，引用计数增加。
     */
    Tensor *tensor_view(const Tensor *src, int ndim, const int *dims, const int *strides);

    /**
     * @brief 根据索引计算元素在数据中的偏移量
     * @param t 张量
     * @param indices 索引数组，长度等于张量维度
     * @return 元素偏移量（以元素个数计），若索引越界返回 SIZE_MAX
     */
    ptrdiff_t tensor_offset(const Tensor *t, const int *indices);

    /* ==================== 数据操作 ==================== */

    /**
     * @brief 将数据从源张量复制到目标张量（要求形状相同）
     * @param dst 目标张量
     * @param src 源张量
     * @return TensorStatus
     * @note 若目标引用计数为1且管理数据，则直接覆盖；否则触发写时拷贝。
     */
    TensorStatus tensor_copy(Tensor *dst, const Tensor *src);

    /**
     * @brief 确保张量数据连续
     * @param t 张量
     * @return TensorStatus
     * @note 若张量已连续，无操作；否则创建新的连续副本（可能改变引用关系）。
     *       若为外部数据（ref_count == NULL），返回 TENSOR_ERR_UNSUPPORTED。
     */
    TensorStatus tensor_contiguous(Tensor *t);

    /* ==================== 写时拷贝机制 ==================== */

    /**
     * @brief 确保张量拥有独占的数据副本（写时拷贝）
     * @param t 目标张量
     * @return TensorStatus
     * @note 若引用计数>1，则复制数据并更新引用计数；若为外部数据（ref_count == NULL），直接返回（假定可写）。
     *       常规修改操作会自动调用此函数。
     */
    TensorStatus tensor_make_unique(Tensor *t);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_TENSOR_H