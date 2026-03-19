#include "tensor.h"
#include "search_ops.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")
#define EPS 1e-5f

static int approx_equal(float a, float b, float eps)
{
    if (isnan(a) && isnan(b))
        return 1;
    if (isinf(a) && isinf(b))
        return (a > 0) == (b > 0);
    return fabsf(a - b) < eps;
}

static int check_tensor(const Tensor *t, const float *expected, size_t n)
{
    if (tensor_size(t) != n)
        return 0;
    int ndim = t->ndim;
    if (ndim == 0)
        return approx_equal(t->data[0], expected[0], EPS);

    int *coords = (int *)malloc(ndim * sizeof(int));
    if (!coords)
        return 0;
    memset(coords, 0, ndim * sizeof(int));

    int strides[TENSOR_MAX_DIM];
    util_get_effective_strides(t, strides);

    size_t idx = 0;
    while (1)
    {
        ptrdiff_t off = 0;
        for (int i = 0; i < ndim; ++i)
            off += (ptrdiff_t)coords[i] * strides[i];
        if (!approx_equal(t->data[off], expected[idx++], EPS))
        {
            free(coords);
            return 0;
        }
        if (util_increment_coords(coords, t->dims, ndim))
            break;
    }
    free(coords);
    return 1;
}

/* ==================== 正常功能测试 ==================== */

void test_sort()
{
    TEST("tensor_sort (axis)");
    int dims[] = {2, 3};
    float data[] = {3, 1, 2, 6, 5, 4};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_sort(src, 1, out);
    float expected[] = {1, 2, 3, 4, 5, 6};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(out);

    out = tensor_create(1, (int[]){6});
    tensor_sort(src, -1, out);
    assert(check_tensor(out, expected, 6));

    tensor_destroy(out);
    tensor_destroy(src);
    PASS();
}

void test_argsort()
{
    TEST("tensor_argsort");
    int dims[] = {2, 3};
    float data[] = {3, 1, 2, 6, 5, 4};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_argsort(src, 1, out);
    float expected[] = {1, 2, 0, 2, 1, 0}; // 每行升序的原始索引
    assert(check_tensor(out, expected, 6));

    tensor_destroy(out);
    tensor_destroy(src);
    PASS();
}

void test_unique()
{
    TEST("tensor_unique");
    int dims[] = {3, 3};
    float data[] = {1, 2, 2, 3, 3, 3, 4, 4, 4};
    Tensor *src = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(1, (int[]){4});

    tensor_unique(src, out);
    float expected[] = {1, 2, 3, 4};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(out);
    tensor_destroy(src);
    PASS();
}

void test_searchsorted()
{
    TEST("tensor_searchsorted");
    int dims_sorted[] = {5};
    float sorted_data[] = {1, 2, 3, 5, 7};
    Tensor *sorted = tensor_from_array(sorted_data, 1, dims_sorted);

    int dims_vals[] = {3};
    float vals_data[] = {2, 4, 8};
    Tensor *values = tensor_from_array(vals_data, 1, dims_vals);

    Tensor *out_left = tensor_create(1, dims_vals);
    tensor_searchsorted(sorted, values, 0, out_left);
    float exp_left[] = {1, 3, 5};
    assert(check_tensor(out_left, exp_left, 3));

    Tensor *out_right = tensor_create(1, dims_vals);
    tensor_searchsorted(sorted, values, 1, out_right);
    float exp_right[] = {2, 3, 5};
    assert(check_tensor(out_right, exp_right, 3));

    tensor_destroy(sorted);
    tensor_destroy(values);
    tensor_destroy(out_left);
    tensor_destroy(out_right);
    PASS();
}

void test_topk()
{
    TEST("tensor_topk");
    int dims[] = {2, 3};
    float data[] = {1, 5, 2, 8, 3, 6};
    Tensor *src = tensor_from_array(data, 2, dims);

    Tensor *vals = tensor_create(2, (int[]){2, 2});
    Tensor *idx = tensor_create(2, (int[]){2, 2});
    tensor_topk(src, 2, 1, 1, 1, vals, idx);

    float exp_vals[] = {5, 2, 8, 6};
    float exp_idx[] = {1, 2, 0, 2};
    assert(check_tensor(vals, exp_vals, 4));
    assert(check_tensor(idx, exp_idx, 4));

    tensor_destroy(vals);
    tensor_destroy(idx);

    vals = tensor_create(2, (int[]){2, 2});
    idx = tensor_create(2, (int[]){2, 2});
    tensor_topk(src, 2, 1, 0, 1, vals, idx);

    float exp_vals_min[] = {1, 2, 3, 6};
    float exp_idx_min[] = {0, 2, 1, 2};
    assert(check_tensor(vals, exp_vals_min, 4));
    assert(check_tensor(idx, exp_idx_min, 4));

    tensor_destroy(vals);
    tensor_destroy(idx);

    vals = tensor_create(1, (int[]){3});
    idx = tensor_create(1, (int[]){3});
    tensor_topk(src, 3, -1, 1, 1, vals, idx);

    float exp_flat_vals[] = {8, 6, 5};
    float exp_flat_idx[] = {3, 5, 1};
    assert(check_tensor(vals, exp_flat_vals, 3));
    assert(check_tensor(idx, exp_flat_idx, 3));

    tensor_destroy(vals);
    tensor_destroy(idx);
    tensor_destroy(src);
    PASS();
}

void test_kthvalue()
{
    TEST("tensor_kthvalue");
    int dims[] = {2, 3};
    float data[] = {1, 5, 2, 8, 3, 6};
    Tensor *src = tensor_from_array(data, 2, dims);

    Tensor *vals = tensor_create(1, (int[]){2});
    Tensor *idx = tensor_create(1, (int[]){2});
    tensor_kthvalue(src, 2, 1, 0, vals, idx);

    float exp_vals[] = {2, 6};
    float exp_idx[] = {2, 2};
    assert(check_tensor(vals, exp_vals, 2));
    assert(check_tensor(idx, exp_idx, 2));

    tensor_destroy(vals);
    tensor_destroy(idx);

    vals = tensor_create(2, (int[]){2, 1});
    idx = tensor_create(2, (int[]){2, 1});
    tensor_kthvalue(src, 2, 1, 1, vals, idx);

    float exp_vals_keep[] = {2, 6};
    float exp_idx_keep[] = {2, 2};
    assert(check_tensor(vals, exp_vals_keep, 2));
    assert(check_tensor(idx, exp_idx_keep, 2));

    tensor_destroy(vals);
    tensor_destroy(idx);

    vals = tensor_create(0, NULL);
    idx = tensor_create(0, NULL);
    tensor_kthvalue(src, 4, -1, 0, vals, idx);

    float exp_flat_val = 5;
    float exp_flat_idx = 1;
    assert(approx_equal(vals->data[0], exp_flat_val, EPS));
    assert(approx_equal(idx->data[0], exp_flat_idx, EPS));

    tensor_destroy(vals);
    tensor_destroy(idx);
    tensor_destroy(src);
    PASS();
}

/* ==================== 错误处理测试 ==================== */

void test_sort_invalid_axis()
{
    TEST("tensor_sort invalid axis");
    int dims[] = {2, 3};
    Tensor *src = tensor_create(2, dims);
    Tensor *out = tensor_create(2, dims);

    TensorStatus st = tensor_sort(src, 2, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    st = tensor_sort(src, -3, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_argsort_invalid_axis()
{
    TEST("tensor_argsort invalid axis");
    int dims[] = {2, 3};
    Tensor *src = tensor_create(2, dims);
    Tensor *out = tensor_create(2, dims);

    TensorStatus st = tensor_argsort(src, 2, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_unique_output_shape_mismatch()
{
    TEST("tensor_unique shape mismatch");
    int dims[] = {3, 3};
    Tensor *src = tensor_create(2, dims);
    Tensor *out = tensor_create(1, (int[]){3}); // 太小

    TensorStatus st = tensor_unique(src, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    tensor_destroy(src);
    tensor_destroy(out);
    PASS();
}

void test_searchsorted_input_not_sorted()
{
    TEST("tensor_searchsorted input not sorted (no crash)");
    int dims_sorted[] = {5};
    float unsorted_data[] = {1, 3, 2, 5, 7};
    Tensor *sorted = tensor_from_array(unsorted_data, 1, dims_sorted);
    int dims_vals[] = {1};
    float vals_data[] = {4};
    Tensor *values = tensor_from_array(vals_data, 1, dims_vals);
    Tensor *out = tensor_create(1, dims_vals);

    tensor_searchsorted(sorted, values, 0, out);

    tensor_destroy(sorted);
    tensor_destroy(values);
    tensor_destroy(out);
    PASS();
}

void test_topk_invalid_k()
{
    TEST("tensor_topk invalid k");
    int dims[] = {2, 3};
    Tensor *src = tensor_create(2, dims);
    Tensor *vals = tensor_create(2, (int[]){2, 2});
    Tensor *idx = tensor_create(2, (int[]){2, 2});

    TensorStatus st = tensor_topk(src, 0, 1, 1, 1, vals, idx);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    st = tensor_topk(src, -1, 1, 1, 1, vals, idx);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    st = tensor_topk(src, 4, 1, 1, 1, vals, idx); // 轴长度3
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(src);
    tensor_destroy(vals);
    tensor_destroy(idx);
    PASS();
}

void test_kthvalue_invalid_k()
{
    TEST("tensor_kthvalue invalid k");
    int dims[] = {2, 3};
    Tensor *src = tensor_create(2, dims);
    Tensor *vals = tensor_create(1, (int[]){2});
    Tensor *idx = tensor_create(1, (int[]){2});

    TensorStatus st = tensor_kthvalue(src, 0, 1, 0, vals, idx);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    st = tensor_kthvalue(src, 4, 1, 0, vals, idx);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(src);
    tensor_destroy(vals);
    tensor_destroy(idx);
    PASS();
}

void test_kthvalue_invalid_axis()
{
    TEST("tensor_kthvalue invalid axis");
    int dims[] = {2, 3};
    Tensor *src = tensor_create(2, dims);
    Tensor *vals = tensor_create(1, (int[]){2});
    Tensor *idx = tensor_create(1, (int[]){2});

    TensorStatus st = tensor_kthvalue(src, 1, 2, 0, vals, idx);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(src);
    tensor_destroy(vals);
    tensor_destroy(idx);
    PASS();
}

/* ==================== 主函数 ==================== */
int main()
{
    test_sort();
    test_argsort();
    test_unique();
    test_searchsorted();
    test_topk();
    test_kthvalue();

    test_sort_invalid_axis();
    test_argsort_invalid_axis();
    test_unique_output_shape_mismatch();
    test_searchsorted_input_not_sorted();
    test_topk_invalid_k();
    test_kthvalue_invalid_k();
    test_kthvalue_invalid_axis();

    printf("All search_ops tests passed!\n");
    return 0;
}