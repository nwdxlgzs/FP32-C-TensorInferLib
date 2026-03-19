#include "tensor.h"
#include "fft_ops.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/**
 * @brief 判断整数是否为 2 的幂
 */
static int is_power_of_two(int x)
{
    return (x > 0) && ((x & (x - 1)) == 0);
}

/**
 * @brief 位反转重排复数数据（仅用于 2 的幂）
 * @param src 输入复数数组（交错实虚），长度 2*n
 * @param dst 输出位反转后的数组（预先分配）
 * @param n 复数个数
 */
static void bit_reverse_copy(const float *src, float *dst, int n)
{
    int logn = 0;
    while ((1 << logn) < n)
        logn++;
    for (int i = 0; i < n; i++)
    {
        int rev = 0;
        for (int j = 0; j < logn; j++)
        {
            if (i & (1 << j))
                rev |= (1 << (logn - 1 - j));
        }
        dst[2 * rev] = src[2 * i];
        dst[2 * rev + 1] = src[2 * i + 1];
    }
}

/**
 * @brief 朴素 DFT（用于非 2 的幂）
 * @param data 复数数组（交错实虚），长度 2*n，结果原地覆盖
 * @param n 复数个数
 * @param inverse 0 正向，非零逆向
 * @return 0 成功，-1 内存错误
 */
static int dft(float *data, int n, int inverse)
{
    float *tmp = (float *)malloc(2 * n * sizeof(float));
    if (!tmp)
        return -1;
    memcpy(tmp, data, 2 * n * sizeof(float));

    float scale = inverse ? 1.0f / n : 1.0f;
    for (int k = 0; k < n; k++)
    {
        float sum_re = 0.0f, sum_im = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float angle = 2 * M_PI * j * k / n;
            float w_re = cosf(angle);
            float w_im = inverse ? sinf(angle) : -sinf(angle);
            float x_re = tmp[2 * j];
            float x_im = tmp[2 * j + 1];
            sum_re += x_re * w_re - x_im * w_im;
            sum_im += x_re * w_im + x_im * w_re;
        }
        data[2 * k] = sum_re * scale;
        data[2 * k + 1] = sum_im * scale;
    }
    free(tmp);
    return 0;
}

/**
 * @brief 基-2 复数 FFT（迭代实现，原地）
 * @param data 复数数组（交错实虚），长度 2*n
 * @param n 复数个数（必须为 2 的幂）
 * @param inverse 0 表示正向 FFT，非零表示逆 FFT（结果除以 n）
 * @return 0 成功，-1 失败
 */
static int fft_radix2(float *data, int n, int inverse)
{
    if (n <= 1)
        return 0;
    float *tmp = (float *)malloc(2 * n * sizeof(float));
    if (!tmp)
        return -1;
    bit_reverse_copy(data, tmp, n);
    memcpy(data, tmp, 2 * n * sizeof(float));
    free(tmp);

    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1)
    {
        float wlen_re, wlen_im;
        if (inverse)
        {
            wlen_re = cosf(2 * M_PI / len);
            wlen_im = sinf(2 * M_PI / len);
        }
        else
        {
            wlen_re = cosf(2 * M_PI / len);
            wlen_im = -sinf(2 * M_PI / len);
        }

        for (int i = 0; i < n; i += len)
        {
            float w_re = 1.0f;
            float w_im = 0.0f;
            for (int j = 0; j < len / 2; j++)
            {
                int i1 = i + j;
                int i2 = i + j + len / 2;

                float u_re = data[2 * i1];
                float u_im = data[2 * i1 + 1];
                float v_re = data[2 * i2] * w_re - data[2 * i2 + 1] * w_im;
                float v_im = data[2 * i2] * w_im + data[2 * i2 + 1] * w_re;

                data[2 * i1] = u_re + v_re;
                data[2 * i1 + 1] = u_im + v_im;
                data[2 * i2] = u_re - v_re;
                data[2 * i2 + 1] = u_im - v_im;

                // 更新旋转因子
                float new_w_re = w_re * wlen_re - w_im * wlen_im;
                float new_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
        }
    }

    // 逆变换时除以 n
    if (inverse)
    {
        float inv_n = 1.0f / n;
        for (int i = 0; i < n; i++)
        {
            data[2 * i] *= inv_n;
            data[2 * i + 1] *= inv_n;
        }
    }
    return 0;
}

/**
 * @brief 通用复数 FFT（自动选择算法）
 * @param data 复数数组（交错实虚），长度 2*n
 * @param n 复数个数
 * @param inverse 0 正向，非零逆向
 * @return 0 成功，-1 失败
 */
static int cfft(float *data, int n, int inverse)
{
    if (n <= 1)
        return 0;
    if (is_power_of_two(n))
        return fft_radix2(data, n, inverse);
    else
        return dft(data, n, inverse);
}

/* ==================== API 实现 ==================== */

TensorStatus tensor_fft_rfft(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 1)
        return TENSOR_ERR_INVALID_PARAM;

    int n = src->dims[src->ndim - 1]; // 信号长度

    // 计算输出形状：[..., n/2+1, 2]  (对于任意 n，n/2+1 即为 floor(n/2)+1)
    int out_ndim = src->ndim + 1;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < src->ndim - 1; i++)
        out_dims[i] = src->dims[i];
    out_dims[src->ndim - 1] = n / 2 + 1;
    out_dims[src->ndim] = 2;

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; i++)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    // 获取步长
    int src_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(out, out_strides);

    // batch 维度：除最后一维外的所有维度
    int batch_ndim = src->ndim - 1;
    int batch_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; i++)
        batch_dims[i] = src->dims[i];

    // 临时复数缓冲区（长度为 n）
    float *complex_buf = (float *)malloc(2 * n * sizeof(float));
    if (!complex_buf)
        return TENSOR_ERR_MEMORY;

    int batch_coords[TENSOR_MAX_DIM] = {0};
    do
    {
        // 计算当前 batch 的基偏移
        size_t src_base = 0, out_base = 0;
        for (int i = 0; i < batch_ndim; i++)
        {
            src_base += batch_coords[i] * src_strides[i];
            out_base += batch_coords[i] * out_strides[i];
        }

        // 将实数填充到复数缓冲区（虚部为 0）
        for (int k = 0; k < n; k++)
        {
            size_t src_off = src_base + k * src_strides[batch_ndim];
            complex_buf[2 * k] = src->data[src_off];
            complex_buf[2 * k + 1] = 0.0f;
        }

        // 正向 FFT（支持任意长度）
        if (cfft(complex_buf, n, 0) != 0)
        {
            free(complex_buf);
            return TENSOR_ERR_MEMORY;
        }
        // 将前 n/2+1 个复数写入输出
        int n_out = n / 2 + 1;
        for (int k = 0; k < n_out; k++)
        {
            size_t out_off_re = out_base + k * out_strides[batch_ndim] + 0 * out_strides[batch_ndim + 1];
            size_t out_off_im = out_base + k * out_strides[batch_ndim] + 1 * out_strides[batch_ndim + 1];
            out->data[out_off_re] = complex_buf[2 * k];
            out->data[out_off_im] = complex_buf[2 * k + 1];
        }

    } while (!util_increment_coords(batch_coords, batch_dims, batch_ndim));

    free(complex_buf);
    return TENSOR_OK;
}

TensorStatus tensor_fft_irfft(const Tensor *src, int n, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 2)
        return TENSOR_ERR_INVALID_PARAM;
    if (src->dims[src->ndim - 1] != 2)
        return TENSOR_ERR_SHAPE_MISMATCH; // 最后一维必须是2

    int m = src->dims[src->ndim - 2]; // 输入的复数个数

    // 确定原始长度 n
    if (n <= 0)
    {
        // 默认推断为偶数长度（兼容旧行为）
        n = 2 * (m - 1);
    }
    // 验证输入的复数个数与 n 是否匹配
    int expected_m;
    if (n % 2 == 0)
        expected_m = n / 2 + 1;
    else
        expected_m = (n + 1) / 2;
    if (m != expected_m)
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 输出形状：[..., n]
    int out_ndim = src->ndim - 1;
    int out_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < out_ndim - 1; i++)
        out_dims[i] = src->dims[i];
    out_dims[out_ndim - 1] = n;

    if (out->ndim != out_ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < out_ndim; i++)
        if (out->dims[i] != out_dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    // 步长
    int src_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(out, out_strides);

    // batch 维度：除最后两维外的所有维度
    int batch_ndim = src->ndim - 2;
    int batch_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; i++)
        batch_dims[i] = src->dims[i];

    float *complex_buf = (float *)malloc(2 * n * sizeof(float));
    if (!complex_buf)
        return TENSOR_ERR_MEMORY;

    int batch_coords[TENSOR_MAX_DIM] = {0};
    do
    {
        size_t src_base = 0, out_base = 0;
        for (int i = 0; i < batch_ndim; i++)
        {
            src_base += batch_coords[i] * src_strides[i];
            out_base += batch_coords[i] * out_strides[i];
        }

        // 读取前 m 个复数
        for (int k = 0; k < m; k++)
        {
            size_t off_re = src_base + k * src_strides[batch_ndim] + 0 * src_strides[batch_ndim + 1];
            size_t off_im = src_base + k * src_strides[batch_ndim] + 1 * src_strides[batch_ndim + 1];
            complex_buf[2 * k] = src->data[off_re];
            complex_buf[2 * k + 1] = src->data[off_im];
        }

        // 利用共轭对称性填充剩余部分
        int k_start = 1;
        int k_end;
        if (n % 2 == 0)
        {
            k_end = m - 2; // 对于偶数，奈奎斯特分量已存在，只需填充 1 .. m-2
        }
        else
        {
            k_end = m - 1; // 对于奇数，填充 1 .. m-1
        }
        for (int k = k_start; k <= k_end; k++)
        {
            int k_sym = n - k;
            complex_buf[2 * k_sym] = complex_buf[2 * k];
            complex_buf[2 * k_sym + 1] = -complex_buf[2 * k + 1];
        }
        // 注意：当 n 为偶数且 m==1 时（n=2），k_end = -1，循环跳过，无需填充

        // 逆 FFT（支持任意长度）
        if (cfft(complex_buf, n, 1) != 0)
        {
            free(complex_buf);
            return TENSOR_ERR_MEMORY;
        }

        // 提取实部到输出（虚部应为零，忽略）
        for (int k = 0; k < n; k++)
        {
            size_t out_off = out_base + k * out_strides[out_ndim - 1];
            out->data[out_off] = complex_buf[2 * k];
        }

    } while (!util_increment_coords(batch_coords, batch_dims, batch_ndim));

    free(complex_buf);
    return TENSOR_OK;
}

// ... 在现有代码末尾添加

TensorStatus tensor_fft(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 2)
        return TENSOR_ERR_INVALID_PARAM;
    if (src->dims[src->ndim - 1] != 2)
        return TENSOR_ERR_SHAPE_MISMATCH; // 最后一维必须是2

    int n = src->dims[src->ndim - 2]; // 信号长度
    int batch_ndim = src->ndim - 2;

    // 检查输出形状是否匹配
    if (out->ndim != src->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < src->ndim; i++)
        if (out->dims[i] != src->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(out, out_strides);

    int batch_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; i++)
        batch_dims[i] = src->dims[i];

    float *complex_buf = (float *)malloc(2 * n * sizeof(float));
    if (!complex_buf)
        return TENSOR_ERR_MEMORY;

    int batch_coords[TENSOR_MAX_DIM] = {0};
    do
    {
        size_t src_base = 0, out_base = 0;
        for (int i = 0; i < batch_ndim; i++)
        {
            src_base += batch_coords[i] * src_strides[i];
            out_base += batch_coords[i] * out_strides[i];
        }

        // 读取复数到缓冲区
        for (int k = 0; k < n; k++)
        {
            size_t off_re = src_base + k * src_strides[batch_ndim] + 0 * src_strides[batch_ndim + 1];
            size_t off_im = src_base + k * src_strides[batch_ndim] + 1 * src_strides[batch_ndim + 1];
            complex_buf[2 * k] = src->data[off_re];
            complex_buf[2 * k + 1] = src->data[off_im];
        }

        // 正向FFT
        if (cfft(complex_buf, n, 0) != 0)
        {
            free(complex_buf);
            return TENSOR_ERR_MEMORY;
        }

        // 写回输出
        for (int k = 0; k < n; k++)
        {
            size_t out_off_re = out_base + k * out_strides[batch_ndim] + 0 * out_strides[batch_ndim + 1];
            size_t out_off_im = out_base + k * out_strides[batch_ndim] + 1 * out_strides[batch_ndim + 1];
            out->data[out_off_re] = complex_buf[2 * k];
            out->data[out_off_im] = complex_buf[2 * k + 1];
        }

    } while (!util_increment_coords(batch_coords, batch_dims, batch_ndim));

    free(complex_buf);
    return TENSOR_OK;
}

TensorStatus tensor_ifft(const Tensor *src, Tensor *out)
{
    if (!src || !out)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 2)
        return TENSOR_ERR_INVALID_PARAM;
    if (src->dims[src->ndim - 1] != 2)
        return TENSOR_ERR_SHAPE_MISMATCH;

    int n = src->dims[src->ndim - 2];
    int batch_ndim = src->ndim - 2;

    if (out->ndim != src->ndim)
        return TENSOR_ERR_SHAPE_MISMATCH;
    for (int i = 0; i < src->ndim; i++)
        if (out->dims[i] != src->dims[i])
            return TENSOR_ERR_SHAPE_MISMATCH;

    TensorStatus status = tensor_make_unique(out);
    if (status != TENSOR_OK)
        return status;

    int src_strides[TENSOR_MAX_DIM], out_strides[TENSOR_MAX_DIM];
    util_get_effective_strides(src, src_strides);
    util_get_effective_strides(out, out_strides);

    int batch_dims[TENSOR_MAX_DIM];
    for (int i = 0; i < batch_ndim; i++)
        batch_dims[i] = src->dims[i];

    float *complex_buf = (float *)malloc(2 * n * sizeof(float));
    if (!complex_buf)
        return TENSOR_ERR_MEMORY;

    int batch_coords[TENSOR_MAX_DIM] = {0};
    do
    {
        size_t src_base = 0, out_base = 0;
        for (int i = 0; i < batch_ndim; i++)
        {
            src_base += batch_coords[i] * src_strides[i];
            out_base += batch_coords[i] * out_strides[i];
        }

        for (int k = 0; k < n; k++)
        {
            size_t off_re = src_base + k * src_strides[batch_ndim] + 0 * src_strides[batch_ndim + 1];
            size_t off_im = src_base + k * src_strides[batch_ndim] + 1 * src_strides[batch_ndim + 1];
            complex_buf[2 * k] = src->data[off_re];
            complex_buf[2 * k + 1] = src->data[off_im];
        }

        // 逆FFT (已除n)
        if (cfft(complex_buf, n, 1) != 0)
        {
            free(complex_buf);
            return TENSOR_ERR_MEMORY;
        }

        for (int k = 0; k < n; k++)
        {
            size_t out_off_re = out_base + k * out_strides[batch_ndim] + 0 * out_strides[batch_ndim + 1];
            size_t out_off_im = out_base + k * out_strides[batch_ndim] + 1 * out_strides[batch_ndim + 1];
            out->data[out_off_re] = complex_buf[2 * k];
            out->data[out_off_im] = complex_buf[2 * k + 1];
        }

    } while (!util_increment_coords(batch_coords, batch_dims, batch_ndim));

    free(complex_buf);
    return TENSOR_OK;
}