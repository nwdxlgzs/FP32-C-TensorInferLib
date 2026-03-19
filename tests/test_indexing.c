#include "tensor.h"
#include "indexing.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")
#define EPS 1e-5f

/* 辅助：比较两个浮点数是否近似相等 */
static int approx_equal(float a, float b, float eps)
{
    if (isnan(a) && isnan(b))
        return 1;
    if (isinf(a) && isinf(b))
        return (a > 0) == (b > 0);
    return fabsf(a - b) < eps;
}

/* 辅助：检查张量所有元素与预期值是否相等 */
static int check_tensor(const Tensor *t, const float *expected, size_t n)
{
    if (tensor_size(t) != n)
        return 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (!approx_equal(t->data[i], expected[i], EPS))
            return 0;
    }
    return 1;
}

/* 辅助：打印张量（调试用） */
static void print_tensor(const Tensor *t, const char *name)
{
    printf("%s: ", name);
    for (size_t i = 0; i < t->size; ++i)
        printf("%.2f ", t->data[i]);
    printf("\n");
}

/* ---------- 测试 tensor_get_item / tensor_set_item ---------- */
void test_get_set_item()
{
    TEST("tensor_get/set_item");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *a = tensor_from_array(data, 2, dims);

    float val;
    TensorStatus status = tensor_get_item(a, (int[]){1, 2}, &val);
    assert(status == TENSOR_OK);
    assert(approx_equal(val, 6.0f, EPS));

    status = tensor_set_item(a, (int[]){0, 1}, 100.0f);
    assert(status == TENSOR_OK);
    assert(approx_equal(a->data[1], 100.0f, EPS));

    // 越界测试
    status = tensor_get_item(a, (int[]){2, 0}, &val);
    assert(status == TENSOR_ERR_INDEX_OUT_OF_BOUNDS);

    tensor_destroy(a);
    PASS();
}

/* ---------- 测试 tensor_slice ---------- */
void test_slice()
{
    TEST("tensor_slice");
    int dims[] = {3, 4};
    float data[12];
    for (int i = 0; i < 12; i++)
        data[i] = (float)i;
    Tensor *a = tensor_from_array(data, 2, dims);

    // 正常切片
    int starts[] = {0, 1};
    int ends[] = {3, 3};
    int steps[] = {1, 2};
    Tensor view = {0};
    TensorStatus status = tensor_slice(a, starts, ends, steps, &view);
    assert(status == TENSOR_OK);

    float expected1[] = {1, 5, 9};
    int idx = 0;
    for (int i = 0; i < view.dims[0]; i++)
    {
        for (int j = 0; j < view.dims[1]; j++)
        {
            ptrdiff_t offset = i * view.strides[0] + j * view.strides[1];
            float val = view.data[offset];
            assert(approx_equal(val, expected1[idx++], EPS));
        }
    }
    tensor_cleanup(&view);

    // 负步长切片
    int starts2[] = {2, 3};
    int ends2[] = {0, 0};
    int steps2[] = {-1, -1};
    Tensor view2 = {0};
    status = tensor_slice(a, starts2, ends2, steps2, &view2);
    assert(status == TENSOR_OK);

    float expected2[] = {11, 10, 9, 7, 6, 5};
    idx = 0;
    for (int i = 0; i < view2.dims[0]; i++)
    {
        for (int j = 0; j < view2.dims[1]; j++)
        {
            ptrdiff_t offset = i * view2.strides[0] + j * view2.strides[1];
            float val = view2.data[offset];
            assert(approx_equal(val, expected2[idx++], EPS));
        }
    }
    tensor_cleanup(&view2);

    tensor_destroy(a);
    PASS();
}

/* ---------- 测试 tensor_advanced_index ---------- */
void test_advanced_index()
{
    TEST("tensor_advanced_index");
    int dims_src[] = {3, 4};
    float data_src[12];
    for (int i = 0; i < 12; i++)
        data_src[i] = (float)i;
    Tensor *src = tensor_from_array(data_src, 2, dims_src);

    // 索引张量：row索引 [0,2] (1D)，col索引 [1,3] (1D) -> 广播形状 (2,)
    float row_data[] = {0, 2};
    float col_data[] = {1, 3};
    Tensor *row_idx = tensor_from_array(row_data, 1, (int[]){2});
    Tensor *col_idx = tensor_from_array(col_data, 1, (int[]){2});
    const Tensor *indices[] = {row_idx, col_idx};

    // 输出形状应为 (2,) + (剩余维度0) = (2,)
    int out_dims[] = {2};
    Tensor *out = tensor_create(1, out_dims);

    tensor_advanced_index(src, indices, 2, out);
    // 预期：src[0,1]=1, src[2,3]=11
    float expected[] = {1, 11};
    assert(check_tensor(out, expected, 2));

    tensor_destroy(row_idx);
    tensor_destroy(col_idx);
    tensor_destroy(out);
    tensor_destroy(src);
    PASS();
}

/* ---------- 测试 tensor_masked_select ---------- */
void test_masked_select()
{
    TEST("tensor_masked_select");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    float mask_data[] = {1, 0, 1, 0, 1, 0};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *mask = tensor_from_array(mask_data, 2, dims);
    int out_dims[] = {3}; // 3个非零
    Tensor *out = tensor_create(1, out_dims);

    tensor_masked_select(src, mask, out);
    float expected[] = {1, 3, 5};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(src);
    tensor_destroy(mask);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 tensor_index_put ---------- */
void test_index_put()
{
    TEST("tensor_index_put");
    int dims_dst[] = {3, 4};
    float data_dst[12] = {0};
    Tensor *dst = tensor_from_array(data_dst, 2, dims_dst);

    // 索引张量：行索引 [1,2] (1D)，列索引 [0,2] (1D) -> 广播形状 (2,)
    float row_data[] = {1, 2};
    float col_data[] = {0, 2};
    Tensor *row_idx = tensor_from_array(row_data, 1, (int[]){2});
    Tensor *col_idx = tensor_from_array(col_data, 1, (int[]){2});
    const Tensor *indices[] = {row_idx, col_idx};

    // values 张量：形状 (2,)，要赋的值
    float val_data[] = {10, 20};
    Tensor *values = tensor_from_array(val_data, 1, (int[]){2});

    tensor_index_put(dst, indices, 2, values);

    // 验证：dst[1,0]=10, dst[2,2]=20
    float expected[12] = {0};
    expected[1 * 4 + 0] = 10;
    expected[2 * 4 + 2] = 20;
    assert(check_tensor(dst, expected, 12));

    tensor_destroy(row_idx);
    tensor_destroy(col_idx);
    tensor_destroy(values);
    tensor_destroy(dst);
    PASS();
}

/* ---------- 测试 tensor_gather ---------- */
void test_gather()
{
    TEST("tensor_gather");
    int dims_src[] = {2, 3};
    float data_src[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(data_src, 2, dims_src);

    // index 张量，沿轴1
    float idx_data[] = {0, 2, 1}; // 形状 (2,?) 实际上需要与 src 形状相同除轴1外，这里取 (2,1) 或 (2,3)? 我们取 (2,1)
    int idx_dims[] = {2, 1};
    Tensor *idx = tensor_from_array(idx_data, 2, idx_dims);

    int out_dims[] = {2, 1};
    Tensor *out = tensor_create(2, out_dims);

    tensor_gather(src, 1, idx, out);
    // 预期：对于行0取列0->1，行1取列1? 但 idx[1,0]=1? 实际idx_data是[0,2,1]但形状(2,1)只用了前两个？我们传入的是[0,2]作为2x1矩阵
    // 因为idx_data只有2个元素，分别是0和2，所以行0列0取0，行1列0取2，结果：src[0,0]=1, src[1,2]=6
    float expected[] = {1, 6};
    assert(check_tensor(out, expected, 2));

    tensor_destroy(src);
    tensor_destroy(idx);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 tensor_scatter ---------- */
void test_scatter()
{
    TEST("tensor_scatter");
    int dims_dst[] = {2, 3};
    float data_dst[6] = {0}; // 全0
    Tensor *dst = tensor_from_array(data_dst, 2, dims_dst);

    // src 张量，形状 (2,1)
    float src_data[] = {10, 20};
    int src_dims[] = {2, 1};
    Tensor *src = tensor_from_array(src_data, 2, src_dims);

    // index 张量，形状 (2,1)，指定要放置的列索引
    float idx_data[] = {1, 2};
    int idx_dims[] = {2, 1};
    Tensor *idx = tensor_from_array(idx_data, 2, idx_dims);

    tensor_scatter(dst, 1, idx, src);

    // 预期：dst[0,1]=10, dst[1,2]=20
    float expected[6] = {0, 10, 0, 0, 0, 20};
    assert(check_tensor(dst, expected, 6));

    tensor_destroy(dst);
    tensor_destroy(src);
    tensor_destroy(idx);
    PASS();
}

void test_take()
{
    TEST("tensor_take");
    int src_dims[] = {2, 3};
    float src_data[] = {1, 2, 3, 4, 5, 6};
    Tensor *src = tensor_from_array(src_data, 2, src_dims);
    int idx_dims[] = {2, 2};
    float idx_data[] = {0, 5, 2, 3};
    Tensor *idx = tensor_from_array(idx_data, 2, idx_dims);
    Tensor *out = tensor_create(2, idx_dims);
    TensorStatus status = tensor_take(src, idx, out);
    assert(status == TENSOR_OK);
    float expected[] = {1, 6, 3, 4};
    assert(check_tensor(out, expected, 4));
    tensor_destroy(src);
    tensor_destroy(idx);
    tensor_destroy(out);
    PASS();
}

void test_put()
{
    TEST("tensor_put");
    int dst_dims[] = {2, 3};
    float dst_data[6] = {0};
    Tensor *dst = tensor_from_array(dst_data, 2, dst_dims);
    int idx_dims[] = {2, 2};
    float idx_data[] = {0, 5, 2, 3};
    Tensor *idx = tensor_from_array(idx_data, 2, idx_dims);
    float val_data[] = {10, 20, 30, 40};
    Tensor *val = tensor_from_array(val_data, 2, idx_dims);
    tensor_put(dst, idx, val, 0);
    float expected1[6] = {10, 0, 30, 40, 0, 20};
    assert(check_tensor(dst, expected1, 6));
    tensor_put(dst, idx, val, 1);
    float expected2[6] = {20, 0, 60, 80, 0, 40};
    assert(check_tensor(dst, expected2, 6));
    tensor_destroy(dst);
    tensor_destroy(idx);
    tensor_destroy(val);
    PASS();
}

void test_nonzero()
{
    TEST("tensor_nonzero");
    int src_dims[] = {2, 3};
    float src_data[] = {0, 2, 0, 3, 0, 5};
    Tensor *src = tensor_from_array(src_data, 2, src_dims);
    int out_dims[] = {3, 2};
    Tensor *out = tensor_create(2, out_dims);
    TensorStatus status = tensor_nonzero(src, out);
    assert(status == TENSOR_OK);
    float expected[] = {0, 1, 1, 0, 1, 2};
    assert(check_tensor(out, expected, 6));
    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

int main()
{
    test_get_set_item();
    test_slice();
    test_advanced_index();
    test_masked_select();
    test_index_put();
    test_gather();
    test_scatter();
    test_take();
    test_put();
    test_nonzero();
    printf("All indexing tests passed!\n");
    return 0;
}