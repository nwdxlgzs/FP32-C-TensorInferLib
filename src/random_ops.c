#include "tensor.h"
#include "random_ops.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ---------- 随机数生成器 ---------- */

static unsigned int g_seed = 1;

void tensor_random_seed(unsigned int seed)
{
    g_seed = seed;
    srand(seed);
}

/* 生成 [0,1) 均匀随机数 */
static float rand_uniform(void)
{
    return (float)rand() / (RAND_MAX + 1.0f);
}

/* ---------- 均匀分布 ---------- */

typedef struct
{
    float low;
    float high;
} uniform_data;

static float gen_uniform(void *data)
{
    uniform_data *ud = (uniform_data *)data;
    return ud->low + (ud->high - ud->low) * rand_uniform();
}

TensorStatus tensor_random_uniform(Tensor *t, float low, float high)
{
    if (high <= low)
        return TENSOR_ERR_INVALID_PARAM;

    uniform_data ud = {low, high};
    return util_generate_op(t, gen_uniform, &ud);
}

/* ---------- 正态分布（Box-Muller） ---------- */

typedef struct
{
    float mean;
    float std;
    int have_spare;
    float spare;
} normal_data;

static float gen_normal(void *data)
{
    normal_data *nd = (normal_data *)data;
    if (nd->have_spare)
    {
        nd->have_spare = 0;
        return nd->mean + nd->std * nd->spare;
    }
    float u1, u2, r, theta;
    do
    {
        u1 = rand_uniform();
    } while (u1 <= FLT_MIN); // 避免 log(0)
    u2 = rand_uniform();
    r = sqrtf(-2.0f * logf(u1));
    theta = 2.0f * (float)M_PI * u2;
    float z0 = r * cosf(theta);
    float z1 = r * sinf(theta);
    nd->spare = z1;
    nd->have_spare = 1;
    return nd->mean + nd->std * z0;
}

TensorStatus tensor_random_normal(Tensor *t, float mean, float std)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    if (std < 0)
        return TENSOR_ERR_INVALID_PARAM;

    normal_data nd = {mean, std, 0, 0.0f};
    return util_generate_op(t, gen_normal, &nd);
}

/* ---------- 截断正态分布（拒绝采样） ---------- */

typedef struct
{
    float mean;
    float std;
    float a; // 原始下界
    float b; // 原始上界
} trunc_normal_data;

static float gen_trunc_normal(void *data)
{
    trunc_normal_data *td = (trunc_normal_data *)data;
    float lower = (td->a - td->mean) / td->std;
    float upper = (td->b - td->mean) / td->std;
    while (1)
    {
        float u1, u2, r, theta, z;
        do
        {
            u1 = rand_uniform();
        } while (u1 <= FLT_MIN);
        u2 = rand_uniform();
        r = sqrtf(-2.0f * logf(u1));
        theta = 2.0f * (float)M_PI * u2;
        z = r * cosf(theta); // 标准正态
        if (z >= lower && z <= upper)
            return td->mean + td->std * z;
    }
}

TensorStatus tensor_random_truncated_normal(Tensor *t, float mean, float std, float a, float b)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    if (std < 0 || a >= b)
        return TENSOR_ERR_INVALID_PARAM;

    trunc_normal_data td = {mean, std, a, b};
    return util_generate_op(t, gen_trunc_normal, &td);
}

/* ---------- 伯努利分布 ---------- */

typedef struct
{
    float p;
} bernoulli_data;

static float gen_bernoulli(void *data)
{
    bernoulli_data *bd = (bernoulli_data *)data;
    return (rand_uniform() < bd->p) ? 1.0f : 0.0f;
}

TensorStatus tensor_random_bernoulli(Tensor *t, float p)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    if (p < 0.0f || p > 1.0f)
        return TENSOR_ERR_INVALID_PARAM;

    bernoulli_data bd = {p};
    return util_generate_op(t, gen_bernoulli, &bd);
}

/* ---------- 随机整数 ---------- */

typedef struct
{
    int low;
    int high;
} randint_data;

static float gen_randint(void *data)
{
    randint_data *rd = (randint_data *)data;
    int range = rd->high - rd->low;
    int r = rand() % range; // 假定 RAND_MAX 足够大，否则可能不均匀，但简单处理
    return (float)(rd->low + r);
}

TensorStatus tensor_random_randint(Tensor *t, int low, int high)
{
    if (!t)
        return TENSOR_ERR_NULL_PTR;
    if (low >= high)
        return TENSOR_ERR_INVALID_PARAM;

    randint_data rd = {low, high};
    return util_generate_op(t, gen_randint, &rd);
}

/* ---------- 打乱（沿第一维） ---------- */

TensorStatus tensor_shuffle(const Tensor *src, Tensor *dst)
{
    if (!src || !dst)
        return TENSOR_ERR_NULL_PTR;
    if (src->ndim < 1)
        return TENSOR_ERR_INVALID_PARAM;
    if (src->ndim != dst->ndim || !util_shapes_equal(src->dims, dst->dims, src->ndim))
        return TENSOR_ERR_SHAPE_MISMATCH;

    // 确保 dst 独占且连续（因为后续使用指针算术）
    TensorStatus status = tensor_contiguous(dst); // tensor_contiguous 已确保独占和连续
    if (status != TENSOR_OK)
        return status;

    // 复制 src 到 dst（利用已修改的 tensor_copy，支持非连续 src）
    status = tensor_copy(dst, src);
    if (status != TENSOR_OK)
        return status;

    // 打乱第一维
    int n = src->dims[0];
    size_t sample_size = src->size / n;
    if (sample_size == 0)
        return TENSOR_OK;

    float *tmp = (float *)malloc(sample_size * sizeof(float));
    if (!tmp)
        return TENSOR_ERR_MEMORY;

    for (int i = 0; i < n - 1; ++i)
    {
        int j = i + rand() / (RAND_MAX / (n - i) + 1);
        if (i == j)
            continue;
        float *p_i = dst->data + i * sample_size;
        float *p_j = dst->data + j * sample_size;
        memcpy(tmp, p_i, sample_size * sizeof(float));
        memcpy(p_i, p_j, sample_size * sizeof(float));
        memcpy(p_j, tmp, sample_size * sizeof(float));
    }
    free(tmp);
    return TENSOR_OK;
}