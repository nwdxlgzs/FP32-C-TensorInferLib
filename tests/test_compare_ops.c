#include "tensor.h"
#include "compare_ops.h"
#include <stdio.h>
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

/* ==================== 原有测试（保持不变） ==================== */

void test_equal()
{
    TEST("tensor_equal");
    int dims[] = {2, 3};
    float data_a[] = {1, 2, 3, 4, 5, 6};
    float data_b[] = {1, 2, 3, 4, 5, 6};
    float data_c[] = {1, 2, 3, 4, 5, 7};
    Tensor *a = tensor_from_array(data_a, 2, dims);
    Tensor *b = tensor_from_array(data_b, 2, dims);
    Tensor *c = tensor_from_array(data_c, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_equal(a, b, out);
    float expected[] = {1, 1, 1, 1, 1, 1};
    assert(check_tensor(out, expected, 6));

    tensor_equal(a, c, out);
    float expected2[] = {1, 1, 1, 1, 1, 0};
    assert(check_tensor(out, expected2, 6));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(c);
    tensor_destroy(out);
    PASS();
}

void test_not_equal()
{
    TEST("tensor_not_equal");
    int dims[] = {3};
    float data_a[] = {1, 2, 3};
    float data_b[] = {1, 0, 3};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_not_equal(a, b, out);
    float expected[] = {0, 1, 0};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_less()
{
    TEST("tensor_less");
    int dims[] = {2, 2};
    float data_a[] = {1, 2, 3, 4};
    float data_b[] = {2, 1, 3, 5};
    Tensor *a = tensor_from_array(data_a, 2, dims);
    Tensor *b = tensor_from_array(data_b, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_less(a, b, out);
    float expected[] = {1, 0, 0, 1}; // 1<2 true, 2<1 false, 3<3 false, 4<5 true
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_less_equal()
{
    TEST("tensor_less_equal");
    int dims[] = {3};
    float data_a[] = {1, 2, 3};
    float data_b[] = {2, 2, 1};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_less_equal(a, b, out);
    float expected[] = {1, 1, 0}; // 1<=2 true, 2<=2 true, 3<=1 false
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_greater()
{
    TEST("tensor_greater");
    int dims[] = {2, 2};
    float data_a[] = {5, 4, 3, 2};
    float data_b[] = {4, 4, 3, 1};
    Tensor *a = tensor_from_array(data_a, 2, dims);
    Tensor *b = tensor_from_array(data_b, 2, dims);
    Tensor *out = tensor_create(2, dims);

    tensor_greater(a, b, out);
    float expected[] = {1, 0, 0, 1};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_greater_equal()
{
    TEST("tensor_greater_equal");
    int dims[] = {3};
    float data_a[] = {1, 2, 3};
    float data_b[] = {1, 1, 4};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_greater_equal(a, b, out);
    float expected[] = {1, 1, 0};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_broadcast()
{
    TEST("tensor_less broadcast");
    int dims_a[] = {2, 1}; // 2x1
    int dims_b[] = {3};    // 3
    float data_a[] = {1, 2};
    float data_b[] = {2, 1, 3};
    Tensor *a = tensor_from_array(data_a, 2, dims_a);
    Tensor *b = tensor_from_array(data_b, 1, dims_b);
    Tensor *out = tensor_create(2, (int[]){2, 3});

    tensor_less(a, b, out);
    // 广播 a: 2x1, b: (3,) -> 2x3
    // row0: 1<2? 1<1? 1<3? -> 1,0,1
    // row1: 2<2? 2<1? 2<3? -> 0,0,1
    float expected[] = {1, 0, 1, 0, 0, 1};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* 原有只测试了 tensor_less_scalar，现在扩展为所有标量比较 */
void test_scalar_ops_old()
{
    TEST("tensor_less_scalar (old)");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_less_scalar(a, 2, out);
    float expected[] = {1, 0, 0};
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* 新增：测试所有标量比较操作 */
void test_all_scalar_compare_ops()
{
    TEST("all scalar compare ops");
    int dims[] = {3};
    float data[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    // equal_scalar
    tensor_equal_scalar(a, 2, out);
    float exp_eq[] = {0, 1, 0};
    assert(check_tensor(out, exp_eq, 3));

    // not_equal_scalar
    tensor_not_equal_scalar(a, 2, out);
    float exp_ne[] = {1, 0, 1};
    assert(check_tensor(out, exp_ne, 3));

    // less_scalar (已测)
    tensor_less_scalar(a, 2, out);
    float exp_lt[] = {1, 0, 0};
    assert(check_tensor(out, exp_lt, 3));

    // less_equal_scalar
    tensor_less_equal_scalar(a, 2, out);
    float exp_le[] = {1, 1, 0};
    assert(check_tensor(out, exp_le, 3));

    // greater_scalar
    tensor_greater_scalar(a, 2, out);
    float exp_gt[] = {0, 0, 1};
    assert(check_tensor(out, exp_gt, 3));

    // greater_equal_scalar
    tensor_greater_equal_scalar(a, 2, out);
    float exp_ge[] = {0, 1, 1};
    assert(check_tensor(out, exp_ge, 3));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* ==================== 新增错误测试 ==================== */

/* 测试形状不兼容的情况 */
void test_shape_mismatch_error()
{
    TEST("compare shape mismatch error");
    int dims_a[] = {2, 3};
    int dims_b[] = {2, 2}; // 不兼容
    Tensor *a = tensor_from_array((float[]){1, 2, 3, 4, 5, 6}, 2, dims_a);
    Tensor *b = tensor_from_array((float[]){1, 2, 3, 4}, 2, dims_b);
    Tensor *out = tensor_create(2, dims_a); // 形状与 a 相同

    TensorStatus st = tensor_equal(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    st = tensor_not_equal(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    st = tensor_less(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    st = tensor_less_equal(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    st = tensor_greater(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    st = tensor_greater_equal(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* 测试输出形状与广播结果不匹配的情况 */
void test_output_shape_mismatch_error()
{
    TEST("compare output shape mismatch error");
    int dims_a[] = {2, 1};
    int dims_b[] = {3};
    Tensor *a = tensor_from_array((float[]){1, 2}, 2, dims_a);
    Tensor *b = tensor_from_array((float[]){2, 1, 3}, 1, dims_b);

    // 广播结果应为 {2,3}，但创建 {2,2}
    Tensor *out_wrong = tensor_create(2, (int[]){2, 2});

    TensorStatus st = tensor_equal(a, b, out_wrong);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out_wrong);
    PASS();
}

/* 测试空指针错误（可选，但确保函数能正确处理） */
void test_null_pointer_error()
{
    TEST("compare null pointer error");
    Tensor *a = tensor_create(1, (int[]){1});
    a->data[0] = 0;
    Tensor *out = tensor_create(1, (int[]){1});

    assert(tensor_equal(NULL, a, out) == TENSOR_ERR_NULL_PTR);
    assert(tensor_equal(a, NULL, out) == TENSOR_ERR_NULL_PTR);
    assert(tensor_equal(a, a, NULL) == TENSOR_ERR_NULL_PTR);

    // 标量版本
    assert(tensor_equal_scalar(NULL, 0, out) == TENSOR_ERR_NULL_PTR);
    assert(tensor_equal_scalar(a, 0, NULL) == TENSOR_ERR_NULL_PTR);

    // 逻辑运算
    assert(tensor_logical_and(NULL, a, out) == TENSOR_ERR_NULL_PTR);
    assert(tensor_logical_not(NULL, out) == TENSOR_ERR_NULL_PTR);

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* 测试逻辑运算 */
void test_logical_and()
{
    TEST("tensor_logical_and");
    int dims[] = {3};
    float data_a[] = {1, 0, 2};
    float data_b[] = {1, 1, 0};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_logical_and(a, b, out);
    float expected[] = {1, 0, 0}; // 1&&1=1, 0&&1=0, 2&&0=0
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_logical_or()
{
    TEST("tensor_logical_or");
    int dims[] = {3};
    float data_a[] = {1, 0, 0};
    float data_b[] = {0, 0, 3};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_logical_or(a, b, out);
    float expected[] = {1, 0, 1}; // 1||0=1, 0||0=0, 0||3=1
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_logical_xor()
{
    TEST("tensor_logical_xor");
    int dims[] = {3};
    float data_a[] = {1, 1, 0};
    float data_b[] = {1, 0, 1};
    Tensor *a = tensor_from_array(data_a, 1, dims);
    Tensor *b = tensor_from_array(data_b, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_logical_xor(a, b, out);
    float expected[] = {0, 1, 1}; // 1^1=0, 1^0=1, 0^1=1
    assert(check_tensor(out, expected, 3));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_logical_not()
{
    TEST("tensor_logical_not");
    int dims[] = {4};
    float data[] = {1, 0, 2, 0};
    Tensor *a = tensor_from_array(data, 1, dims);
    Tensor *out = tensor_create(1, dims);

    tensor_logical_not(a, out);
    float expected[] = {0, 1, 0, 1};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* 测试广播情况下的逻辑运算 */
void test_logical_broadcast()
{
    TEST("logical ops broadcast");
    int dims_a[] = {2, 1};
    int dims_b[] = {3};
    float data_a[] = {1, 0};
    float data_b[] = {1, 0, 1};
    Tensor *a = tensor_from_array(data_a, 2, dims_a);
    Tensor *b = tensor_from_array(data_b, 1, dims_b);
    Tensor *out = tensor_create(2, (int[]){2, 3});

    tensor_logical_and(a, b, out);
    // row0: 1 && [1,0,1] -> [1,0,1]
    // row1: 0 && [1,0,1] -> [0,0,0]
    float exp_and[] = {1, 0, 1, 0, 0, 0};
    assert(check_tensor(out, exp_and, 6));

    tensor_logical_or(a, b, out);
    // row0: 1 || [1,0,1] -> [1,1,1]
    // row1: 0 || [1,0,1] -> [1,0,1]
    float exp_or[] = {1, 1, 1, 1, 0, 1};
    assert(check_tensor(out, exp_or, 6));

    tensor_logical_xor(a, b, out);
    // row0: 1 xor [1,0,1] -> [0,1,0]
    // row1: 0 xor [1,0,1] -> [1,0,1]
    float exp_xor[] = {0, 1, 0, 1, 0, 1};
    assert(check_tensor(out, exp_xor, 6));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ==================== 主函数 ==================== */

int main()
{
    // 原有基础测试
    test_equal();
    test_not_equal();
    test_less();
    test_less_equal();
    test_greater();
    test_greater_equal();
    test_broadcast();
    test_scalar_ops_old();         // 原有只测了 less_scalar
    test_all_scalar_compare_ops(); // 完整覆盖所有标量比较

    // 逻辑运算测试
    test_logical_and();
    test_logical_or();
    test_logical_xor();
    test_logical_not();
    test_logical_broadcast();

    // 错误处理测试
    test_shape_mismatch_error();
    test_output_shape_mismatch_error();
    test_null_pointer_error();

    printf("All compare_ops tests passed!\n");
    return 0;
}