#include "tensor.h"
#include "linalg_ops.h"
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

/* ---------- 测试矩阵乘法 ---------- */
void test_matmul_2d()
{
    TEST("tensor_matmul 2D");
    int dims_a[] = {2, 3};
    int dims_b[] = {3, 4};
    float data_a[] = {1, 2, 3, 4, 5, 6};
    float data_b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor *a = tensor_from_array(data_a, 2, dims_a);
    Tensor *b = tensor_from_array(data_b, 2, dims_b);
    Tensor *out = tensor_create(2, (int[]){2, 4});

    tensor_matmul(a, b, out);
    float expected[] = {
        1 * 1 + 2 * 5 + 3 * 9, 1 * 2 + 2 * 6 + 3 * 10, 1 * 3 + 2 * 7 + 3 * 11, 1 * 4 + 2 * 8 + 3 * 12,
        4 * 1 + 5 * 5 + 6 * 9, 4 * 2 + 5 * 6 + 6 * 10, 4 * 3 + 5 * 7 + 6 * 11, 4 * 4 + 5 * 8 + 6 * 12};
    assert(check_tensor(out, expected, 8));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_matmul_batch()
{
    TEST("tensor_matmul batch broadcast");
    /* a: (2,1,2,3)  b: (1,2,3,4) 广播为 (2,2,2,4) */
    int dims_a[] = {2, 1, 2, 3};
    int dims_b[] = {1, 2, 3, 4};
    float data_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // 2*1*2*3
    float data_b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    Tensor *a = tensor_from_array(data_a, 4, dims_a);
    Tensor *b = tensor_from_array(data_b, 4, dims_b);
    Tensor *out = tensor_create(4, (int[]){2, 2, 2, 4});

    tensor_matmul(a, b, out);

    // 预计算四个块的正确结果
    // 块 (0,0): A0 * B0
    float expected_00[] = {38, 44, 50, 56, 83, 98, 113, 128};
    // 块 (0,1): A0 * B1
    float expected_01[] = {110, 116, 122, 128, 263, 278, 293, 308};
    // 块 (1,0): A1 * B0
    float expected_10[] = {128, 152, 176, 200, 173, 206, 239, 272};
    // 块 (1,1): A1 * B1
    float expected_11[] = {416, 440, 464, 488, 569, 602, 635, 668};

    // 遍历所有输出元素，按顺序 (i_batch, j_batch, row, col)
    size_t idx = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            float *exp;
            if (i == 0 && j == 0)
                exp = expected_00;
            else if (i == 0 && j == 1)
                exp = expected_01;
            else if (i == 1 && j == 0)
                exp = expected_10;
            else
                exp = expected_11;

            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    if (!approx_equal(out->data[idx++], exp[r * 4 + c], EPS))
                    {
                        printf("Mismatch at i=%d j=%d r=%d c=%d: got %f expected %f\n",
                               i, j, r, c, out->data[idx - 1], exp[r * 4 + c]);
                        assert(0);
                    }
                }
            }
        }
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试批量矩阵乘法 ---------- */
void test_bmm()
{
    TEST("tensor_bmm");
    int dims_a[] = {2, 2, 3};
    int dims_b[] = {2, 3, 4};
    float data_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float data_b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    Tensor *a = tensor_from_array(data_a, 3, dims_a);
    Tensor *b = tensor_from_array(data_b, 3, dims_b);
    Tensor *out = tensor_create(3, (int[]){2, 2, 4});

    tensor_bmm(a, b, out);

    /* 批次0: a[0]=[[1,2,3],[4,5,6]], b[0]=[[1,2,3,4],[5,6,7,8],[9,10,11,12]] */
    float expected0[] = {38, 44, 50, 56, 83, 98, 113, 128};
    /* 批次1: a[1]=[[7,8,9],[10,11,12]], b[1]=[[13,14,15,16],[17,18,19,20],[21,22,23,24]] */
    float expected1[] = {416, 440, 464, 488, 569, 602, 635, 668};

    for (int i = 0; i < 2; i++)
    {
        float *exp = (i == 0) ? expected0 : expected1;
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                assert(approx_equal(out->data[i * 2 * 4 + j * 4 + k], exp[j * 4 + k], EPS));
            }
        }
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试点积 ---------- */
void test_dot()
{
    TEST("tensor_dot");
    float data_a[] = {1, 2, 3};
    float data_b[] = {4, 5, 6};
    Tensor *a = tensor_from_array(data_a, 1, (int[]){3});
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    Tensor *out = tensor_create(0, NULL); // 标量

    tensor_dot(a, b, out);
    assert(approx_equal(out->data[0], 1 * 4 + 2 * 5 + 3 * 6, EPS));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试外积 ---------- */
void test_outer()
{
    TEST("tensor_outer");
    float data_a[] = {1, 2};
    float data_b[] = {3, 4, 5};
    Tensor *a = tensor_from_array(data_a, 1, (int[]){2});
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    Tensor *out = tensor_create(2, (int[]){2, 3});

    tensor_outer(a, b, out);
    float expected[] = {1 * 3, 1 * 4, 1 * 5, 2 * 3, 2 * 4, 2 * 5};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试张量缩并 ---------- */
void test_tensordot()
{
    TEST("tensor_tensordot (single-axis)");
    /* 等价于矩阵乘法：a (2,3) b (3,4)，缩并最后一轴 */
    int dims_a[] = {2, 3};
    int dims_b[] = {3, 4};
    float data_a[] = {1, 2, 3, 4, 5, 6};
    float data_b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor *a = tensor_from_array(data_a, 2, dims_a);
    Tensor *b = tensor_from_array(data_b, 2, dims_b);
    Tensor *out = tensor_create(2, (int[]){2, 4});

    int axes_a[] = {1};
    int axes_b[] = {0};
    tensor_tensordot(a, b, axes_a, axes_b, 1, out);

    float expected[] = {38, 44, 50, 56, 83, 98, 113, 128};
    assert(check_tensor(out, expected, 8));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试转置 ---------- */
void test_transpose()
{
    TEST("tensor_transpose");
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *a = tensor_from_array(data, 2, dims);
    Tensor *out = tensor_create(2, (int[]){3, 2});

    tensor_transpose(a, out);
    float expected[] = {1, 4, 2, 5, 3, 6};
    assert(check_tensor(out, expected, 6));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试一般转置 ---------- */
void test_permute()
{
    TEST("tensor_permute");
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++)
        data[i] = (float)i;
    Tensor *a = tensor_from_array(data, 3, dims);
    /* 将轴顺序 (0,2,1) -> 新形状 (2,4,3) */
    int axes[] = {0, 2, 1};
    Tensor *out = tensor_create(3, (int[]){2, 4, 3});

    tensor_permute(a, axes, out);

    /* 验证几个点：原坐标 (1,2,3) 应映射到新坐标 (1,3,2) */
    size_t off_old = 1 * 12 + 2 * 4 + 3;
    size_t off_new = 1 * 12 + 3 * 3 + 2;
    assert(out->data[off_new] == a->data[off_old]);

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试 diag ---------- */
void test_diag_1d()
{
    TEST("tensor_diag (1D -> 2D)");
    float data[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data, 1, (int[]){3});
    Tensor *out = tensor_create(2, (int[]){3, 3});

    tensor_diag(a, out);
    float expected[] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    assert(check_tensor(out, expected, 9));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

void test_diag_2d()
{
    TEST("tensor_diag (2D -> 1D)");
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *a = tensor_from_array(data, 2, (int[]){2, 3});
    Tensor *out = tensor_create(1, (int[]){2});

    tensor_diag(a, out);
    float expected[] = {1, 5}; // 对角线 [0,0] 和 [1,1]
    assert(check_tensor(out, expected, 2));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试迹 ---------- */
void test_trace()
{
    TEST("tensor_trace");
    int dims[] = {2, 3, 3}; // 2个3x3矩阵
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    Tensor *a = tensor_from_array(data, 3, dims);
    Tensor *out = tensor_create(1, (int[]){2}); // 沿轴1和2求迹，保留批次

    tensor_trace(a, 1, 2, out);
    float expected[] = {1 + 5 + 9, 10 + 14 + 18}; // 各矩阵的迹
    assert(check_tensor(out, expected, 2));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* ---------- 测试矩阵求逆 ---------- */
void test_inv_2x2()
{
    TEST("tensor_inv 2x2");
    float data[] = {4, 7, 2, 6}; // 行列式 = 4*6 - 7*2 = 24-14=10
    Tensor *a = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *out = tensor_create(2, (int[]){2, 2});

    tensor_inv(a, out);
    // 逆 = (1/10) * [6, -7; -2, 4]
    float expected[] = {0.6f, -0.7f, -0.2f, 0.4f};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}
void test_inv_identity()
{
    TEST("tensor_inv identity");
    float data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Tensor *a = tensor_from_array(data, 2, (int[]){3, 3});
    Tensor *out = tensor_create(2, (int[]){3, 3});

    tensor_inv(a, out);
    assert(check_tensor(out, data, 9));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}
void test_inv_3x3()
{
    TEST("tensor_inv 3x3");
    float data[] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
    Tensor *a = tensor_from_array(data, 2, (int[]){3, 3});
    Tensor *out = tensor_create(2, (int[]){3, 3});

    tensor_inv(a, out);
    float expected[] = {
        -24, 18, 5,
        20, -15, -4,
        -5, 4, 1};
    printf("\nActual inverse: ");
    for (int i = 0; i < 9; i++)
        printf("%f ", out->data[i]);
    printf("\nExpected: -24 18 5 20 -15 -4 -5 4 1\n");
    assert(check_tensor(out, expected, 9));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}
/* ---------- 新增：测试 tensor_matmul 的一维情形 ---------- */

void test_matmul_1d_dot()
{
    TEST("tensor_matmul 1D dot");
    float data_a[] = {1, 2, 3};
    float data_b[] = {4, 5, 6};
    Tensor *a = tensor_from_array(data_a, 1, (int[]){3});
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    Tensor *out = tensor_create(0, NULL); // 标量

    tensor_matmul(a, b, out);
    assert(approx_equal(out->data[0], 1 * 4 + 2 * 5 + 3 * 6, EPS));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_matmul_vector_matrix()
{
    TEST("tensor_matmul vector * matrix");
    // a: (3,) , b: (2,3,4)  -> 输出 (2,4)
    float data_a[] = {1, 2, 3};
    int dims_a[] = {3};
    float data_b[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,         // batch0
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 // batch1
    };
    int dims_b[] = {2, 3, 4};
    Tensor *a = tensor_from_array(data_a, 1, dims_a);
    Tensor *b = tensor_from_array(data_b, 3, dims_b);
    int out_dims[] = {2, 4};
    Tensor *out = tensor_create(2, out_dims);

    tensor_matmul(a, b, out);
    // 预期: batch0: [38,44,50,56]; batch1: [110,116,122,128]
    float expected[] = {38, 44, 50, 56, 110, 116, 122, 128};
    assert(check_tensor(out, expected, 8));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_matmul_matrix_vector()
{
    TEST("tensor_matmul matrix * vector");
    // a: (2,3) , b: (3,)
    float data_a[] = {1, 2, 3, 4, 5, 6};
    int dims_a[] = {2, 3};
    float data_b[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data_a, 2, dims_a);
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    int out_dims[] = {2};
    Tensor *out = tensor_create(1, out_dims);

    tensor_matmul(a, b, out);
    float expected[] = {14, 32};
    assert(check_tensor(out, expected, 2));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_matmul_batch_matrix_vector()
{
    TEST("tensor_matmul batch matrix * vector");
    // a: (2,2,3) , b: (3,)
    float data_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // 2x2x3
    int dims_a[] = {2, 2, 3};
    float data_b[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data_a, 3, dims_a);
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    int out_dims[] = {2, 2};
    Tensor *out = tensor_create(2, out_dims);

    tensor_matmul(a, b, out);
    // 预期: batch0: [14,32]; batch1: [50,68]
    float expected[] = {14, 32, 50, 68};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}
/* ---------- 主函数 ---------- */
int main()
{
    test_matmul_2d();
    test_matmul_batch();
    test_bmm();
    test_dot();
    test_outer();
    test_tensordot();
    test_transpose();
    test_permute();
    test_diag_1d();
    test_diag_2d();
    test_trace();
    test_inv_2x2();
    test_inv_identity();
    test_inv_3x3();
    test_matmul_1d_dot();
    test_matmul_vector_matrix();
    test_matmul_matrix_vector();
    test_matmul_batch_matrix_vector();

    printf("All linalg_ops tests passed!\n");
    return 0;
}