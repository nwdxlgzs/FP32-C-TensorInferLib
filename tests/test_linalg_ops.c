#include "tensor.h"
#include "linalg_ops.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

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

/* ---------- 原有测试 ---------- */

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

void test_dot()
{
    TEST("tensor_dot");
    float data_a[] = {1, 2, 3};
    float data_b[] = {4, 5, 6};
    Tensor *a = tensor_from_array(data_a, 1, (int[]){3});
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    Tensor *out = tensor_create(0, NULL);

    tensor_dot(a, b, out);
    assert(approx_equal(out->data[0], 1 * 4 + 2 * 5 + 3 * 6, EPS));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

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

void test_permute()
{
    TEST("tensor_permute");
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++)
        data[i] = (float)i;
    Tensor *a = tensor_from_array(data, 3, dims);
    int axes[] = {0, 2, 1};
    Tensor *out = tensor_create(3, (int[]){2, 4, 3});

    tensor_permute(a, axes, out);

    size_t off_old = 1 * 12 + 2 * 4 + 3;
    size_t off_new = 1 * 12 + 3 * 3 + 2;
    assert(out->data[off_new] == a->data[off_old]);

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

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
    float expected[] = {1, 5};
    assert(check_tensor(out, expected, 2));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

void test_trace()
{
    TEST("tensor_trace");
    int dims[] = {2, 3, 3};
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    Tensor *a = tensor_from_array(data, 3, dims);
    Tensor *out = tensor_create(1, (int[]){2});

    tensor_trace(a, 1, 2, out);
    float expected[] = {1 + 5 + 9, 10 + 14 + 18};
    assert(check_tensor(out, expected, 2));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

void test_inv_2x2()
{
    TEST("tensor_inv 2x2");
    float data[] = {4, 7, 2, 6};
    Tensor *a = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *out = tensor_create(2, (int[]){2, 2});

    tensor_inv(a, out);
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
    assert(check_tensor(out, expected, 9));

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

void test_matmul_1d_dot()
{
    TEST("tensor_matmul 1D dot");
    float data_a[] = {1, 2, 3};
    float data_b[] = {4, 5, 6};
    Tensor *a = tensor_from_array(data_a, 1, (int[]){3});
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    Tensor *out = tensor_create(0, NULL);

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
    float data_a[] = {1, 2, 3};
    int dims_a[] = {3};
    float data_b[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    int dims_b[] = {2, 3, 4};
    Tensor *a = tensor_from_array(data_a, 1, dims_a);
    Tensor *b = tensor_from_array(data_b, 3, dims_b);
    int out_dims[] = {2, 4};
    Tensor *out = tensor_create(2, out_dims);

    tensor_matmul(a, b, out);
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
    float data_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int dims_a[] = {2, 2, 3};
    float data_b[] = {1, 2, 3};
    Tensor *a = tensor_from_array(data_a, 3, dims_a);
    Tensor *b = tensor_from_array(data_b, 1, (int[]){3});
    int out_dims[] = {2, 2};
    Tensor *out = tensor_create(2, out_dims);

    tensor_matmul(a, b, out);
    float expected[] = {14, 32, 50, 68};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_det_2x2()
{
    TEST("tensor_det 2x2");
    float data[] = {4, 7, 2, 6};
    Tensor *a = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *out = tensor_create(0, NULL);
    tensor_det(a, out);
    assert(approx_equal(out->data[0], 4 * 6 - 7 * 2, EPS));
    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

void test_det_3x3()
{
    TEST("tensor_det 3x3");
    float data[] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
    Tensor *a = tensor_from_array(data, 2, (int[]){3, 3});
    Tensor *out = tensor_create(0, NULL);
    tensor_det(a, out);
    assert(approx_equal(out->data[0], 1.0f, EPS));
    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

void test_solve_2x2()
{
    TEST("tensor_solve 2x2");
    float data_A[] = {4, 7, 2, 6};
    float data_B[] = {1, 2};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){2, 2});
    Tensor *B = tensor_from_array(data_B, 1, (int[]){2});
    Tensor *X = tensor_create(1, (int[]){2});
    tensor_solve(A, B, X);
    float expected[] = {-0.8f, 0.6f};
    assert(check_tensor(X, expected, 2));
    tensor_destroy(A);
    tensor_destroy(B);
    tensor_destroy(X);
    PASS();
}

void test_solve_3x3()
{
    TEST("tensor_solve 3x3");
    float data_A[] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
    float data_B[] = {1, 0, 1};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){3, 3});
    Tensor *B = tensor_from_array(data_B, 1, (int[]){3});
    Tensor *X = tensor_create(1, (int[]){3});
    tensor_solve(A, B, X);
    float expected[] = {-19.0f, 16.0f, -4.0f};
    assert(check_tensor(X, expected, 3));
    tensor_destroy(A);
    tensor_destroy(B);
    tensor_destroy(X);
    PASS();
}

void test_cholesky_3x3()
{
    TEST("tensor_cholesky 3x3");
    float data_A[] = {4, 12, -16, 12, 37, -43, -16, -43, 98};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){3, 3});
    Tensor *L = tensor_create(2, (int[]){3, 3});
    tensor_cholesky(A, L);
    float expected[] = {2, 0, 0, 6, 1, 0, -8, 5, 3};
    assert(check_tensor(L, expected, 9));
    tensor_destroy(A);
    tensor_destroy(L);
    PASS();
}

void test_qr_3x2_reduced()
{
    TEST("tensor_qr reduced (3x2)");
    float data_A[] = {1, 2, 3, 4, 5, 6};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){3, 2});

    Tensor *Q = tensor_create(2, (int[]){3, 2});
    Tensor *R = tensor_create(2, (int[]){2, 2});
    tensor_qr(A, Q, R, 1);

    Tensor *QR = tensor_create(2, (int[]){3, 2});
    tensor_matmul(Q, R, QR);
    assert(tensor_allclose(A, QR, EPS, EPS));

    Tensor *QT = tensor_create(2, (int[]){2, 3});
    tensor_permute(Q, (int[]){1, 0}, QT);
    Tensor *QTQ = tensor_create(2, (int[]){2, 2});
    tensor_matmul(QT, Q, QTQ);
    float I2[] = {1, 0, 0, 1};
    assert(check_tensor(QTQ, I2, 4));

    for (int i = 1; i < 2; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            assert(approx_equal(R->data[i * 2 + j], 0.0f, EPS));
        }
    }

    tensor_destroy(A);
    tensor_destroy(Q);
    tensor_destroy(R);
    tensor_destroy(QR);
    tensor_destroy(QT);
    tensor_destroy(QTQ);
    PASS();
}

void test_qr_3x2_full()
{
    TEST("tensor_qr full (3x2)");
    float data_A[] = {1, 2, 3, 4, 5, 6};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){3, 2});

    Tensor *Q = tensor_create(2, (int[]){3, 3});
    Tensor *R = tensor_create(2, (int[]){3, 2});
    tensor_qr(A, Q, R, 0);

    Tensor *QR = tensor_create(2, (int[]){3, 2});
    tensor_matmul(Q, R, QR);
    assert(tensor_allclose(A, QR, EPS, EPS));

    Tensor *QT = tensor_create(2, (int[]){3, 3});
    tensor_permute(Q, (int[]){1, 0}, QT);
    Tensor *QTQ = tensor_create(2, (int[]){3, 3});
    tensor_matmul(QT, Q, QTQ);
    float I3[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    assert(check_tensor(QTQ, I3, 9));

    assert(approx_equal(R->data[1 * 2 + 0], 0.0f, EPS));
    for (int j = 0; j < 2; ++j)
    {
        assert(approx_equal(R->data[2 * 2 + j], 0.0f, EPS));
    }

    tensor_destroy(A);
    tensor_destroy(Q);
    tensor_destroy(R);
    tensor_destroy(QR);
    tensor_destroy(QT);
    tensor_destroy(QTQ);
    PASS();
}

void test_svd_3x2_reduced()
{
    TEST("tensor_svd reduced (3x2)");
    float data_A[] = {1, 2, 3, 4, 5, 6};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){3, 2});
    Tensor *U = tensor_create(2, (int[]){3, 2});
    Tensor *S = tensor_create(1, (int[]){2});
    Tensor *V = tensor_create(2, (int[]){2, 2});
    tensor_svd(A, U, S, V, 0);

    Tensor *US = tensor_create(2, (int[]){3, 2});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            US->data[i * 2 + j] = U->data[i * 2 + j] * S->data[j];
    Tensor *VT = tensor_create(2, (int[]){2, 2});
    tensor_permute(V, (int[]){1, 0}, VT);
    Tensor *A_recon = tensor_create(2, (int[]){3, 2});
    tensor_matmul(US, VT, A_recon);
    assert(tensor_allclose(A, A_recon, EPS, EPS));

    Tensor *UT = tensor_create(2, (int[]){2, 3});
    tensor_permute(U, (int[]){1, 0}, UT);
    Tensor *UTU = tensor_create(2, (int[]){2, 2});
    tensor_matmul(UT, U, UTU);
    float I2[] = {1, 0, 0, 1};
    assert(check_tensor(UTU, I2, 4));

    Tensor *VTV = tensor_create(2, (int[]){2, 2});
    Tensor *VT2 = tensor_create(2, (int[]){2, 2});
    tensor_permute(V, (int[]){1, 0}, VT2);
    tensor_matmul(VT2, V, VTV);
    assert(check_tensor(VTV, I2, 4));

    tensor_destroy(A);
    tensor_destroy(U);
    tensor_destroy(S);
    tensor_destroy(V);
    tensor_destroy(US);
    tensor_destroy(VT);
    tensor_destroy(A_recon);
    tensor_destroy(UT);
    tensor_destroy(UTU);
    tensor_destroy(VT2);
    tensor_destroy(VTV);
    PASS();
}

void test_svd_3x2_full()
{
    TEST("tensor_svd full (3x2)");
    float data_A[] = {1, 2, 3, 4, 5, 6};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){3, 2});
    Tensor *U = tensor_create(2, (int[]){3, 3});
    Tensor *S = tensor_create(1, (int[]){2});
    Tensor *V = tensor_create(2, (int[]){2, 2});
    tensor_svd(A, U, S, V, 1);

    Tensor *U_reduced = tensor_create(2, (int[]){3, 2});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            U_reduced->data[i * 2 + j] = U->data[i * 3 + j];

    Tensor *US = tensor_create(2, (int[]){3, 2});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            US->data[i * 2 + j] = U_reduced->data[i * 2 + j] * S->data[j];
    Tensor *VT = tensor_create(2, (int[]){2, 2});
    tensor_permute(V, (int[]){1, 0}, VT);
    Tensor *A_recon = tensor_create(2, (int[]){3, 2});
    tensor_matmul(US, VT, A_recon);
    assert(tensor_allclose(A, A_recon, EPS, EPS));

    Tensor *UT = tensor_create(2, (int[]){3, 3});
    tensor_permute(U, (int[]){1, 0}, UT);
    Tensor *UTU = tensor_create(2, (int[]){3, 3});
    tensor_matmul(UT, U, UTU);
    float I3[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    assert(check_tensor(UTU, I3, 9));

    Tensor *VTV = tensor_create(2, (int[]){2, 2});
    Tensor *VT2 = tensor_create(2, (int[]){2, 2});
    tensor_permute(V, (int[]){1, 0}, VT2);
    tensor_matmul(VT2, V, VTV);
    float I2[] = {1, 0, 0, 1};
    assert(check_tensor(VTV, I2, 4));

    tensor_destroy(A);
    tensor_destroy(U);
    tensor_destroy(S);
    tensor_destroy(V);
    tensor_destroy(U_reduced);
    tensor_destroy(US);
    tensor_destroy(VT);
    tensor_destroy(A_recon);
    tensor_destroy(UT);
    tensor_destroy(UTU);
    tensor_destroy(VT2);
    tensor_destroy(VTV);
    PASS();
}

void test_eigh_3x3()
{
    TEST("tensor_eigh 3x3");
    float data_A[] = {4, 1, 2,
                      1, 3, 0,
                      2, 0, 5};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){3, 3});
    Tensor *eigvals = tensor_create(1, (int[]){3});
    Tensor *eigvecs = tensor_create(2, (int[]){3, 3});

    tensor_eigh(A, eigvals, eigvecs);

    Tensor *AV = tensor_create(2, (int[]){3, 3});
    tensor_matmul(A, eigvecs, AV);

    Tensor *diag = tensor_create(2, (int[]){3, 3});
    tensor_diag(eigvals, diag);
    Tensor *Vdiag = tensor_create(2, (int[]){3, 3});
    tensor_matmul(eigvecs, diag, Vdiag);

    assert(tensor_allclose(AV, Vdiag, EPS, EPS));

    Tensor *VT = tensor_create(2, (int[]){3, 3});
    tensor_permute(eigvecs, (int[]){1, 0}, VT);
    Tensor *VTV = tensor_create(2, (int[]){3, 3});
    tensor_matmul(VT, eigvecs, VTV);
    float I3[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    assert(check_tensor(VTV, I3, 9));

    tensor_destroy(A);
    tensor_destroy(eigvals);
    tensor_destroy(eigvecs);
    tensor_destroy(AV);
    tensor_destroy(diag);
    tensor_destroy(Vdiag);
    tensor_destroy(VT);
    tensor_destroy(VTV);
    PASS();
}

/* ---------- 新增错误测试 ---------- */

/* 奇异矩阵求逆应返回错误 */
void test_inv_singular()
{
    TEST("tensor_inv singular");
    float data[] = {1, 2, 2, 4}; // 行列式 = 0
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *out = tensor_create(2, (int[]){2, 2});
    TensorStatus st = tensor_inv(A, out);
    assert(st == TENSOR_ERR_DIV_BY_ZERO || st == TENSOR_ERR_SINGULAR_MATRIX);
    tensor_destroy(A);
    tensor_destroy(out);
    PASS();
}

/* 奇异矩阵行列式应返回错误且输出0 */
void test_det_singular()
{
    TEST("tensor_det singular");
    float data[] = {1, 2, 2, 4};
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *out = tensor_create(0, NULL);
    TensorStatus st = tensor_det(A, out);
    assert(st == TENSOR_ERR_SINGULAR_MATRIX);
    assert(approx_equal(out->data[0], 0.0f, EPS));
    tensor_destroy(A);
    tensor_destroy(out);
    PASS();
}

/* 线性方程组求解奇异矩阵返回错误 */
void test_solve_singular()
{
    TEST("tensor_solve singular");
    float data_A[] = {1, 2, 2, 4};
    float data_B[] = {1, 2};
    Tensor *A = tensor_from_array(data_A, 2, (int[]){2, 2});
    Tensor *B = tensor_from_array(data_B, 1, (int[]){2});
    Tensor *X = tensor_create(1, (int[]){2});
    TensorStatus st = tensor_solve(A, B, X);
    assert(st == TENSOR_ERR_DIV_BY_ZERO);
    tensor_destroy(A);
    tensor_destroy(B);
    tensor_destroy(X);
    PASS();
}

/* Cholesky 对非正定矩阵返回错误 */
void test_cholesky_non_positive_definite()
{
    TEST("tensor_cholesky non-PD");
    float data[] = {1, 2, 2, 1}; // 不是正定（行列式<0）
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *L = tensor_create(2, (int[]){2, 2});
    TensorStatus st = tensor_cholesky(A, L);
    assert(st == TENSOR_ERR_INVALID_PARAM); // 或别的错误码，根据实现
    tensor_destroy(A);
    tensor_destroy(L);
    PASS();
}

/* 秩亏矩阵的 QR 分解测试 (2x2 全1矩阵) */
void test_qr_rank_deficient()
{
    TEST("tensor_qr rank deficient");
    float data[] = {1, 1, 1, 1};
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});

    Tensor *Q = tensor_create(2, (int[]){2, 2});
    Tensor *R = tensor_create(2, (int[]){2, 2});
    tensor_qr(A, Q, R, 0); // 完全分解

    Tensor *QR = tensor_create(2, (int[]){2, 2});
    tensor_matmul(Q, R, QR);
    assert(tensor_allclose(A, QR, EPS, EPS));

    Tensor *QT = tensor_create(2, (int[]){2, 2});
    tensor_permute(Q, (int[]){1, 0}, QT);
    Tensor *QTQ = tensor_create(2, (int[]){2, 2});
    tensor_matmul(QT, Q, QTQ);
    float I2[] = {1, 0, 0, 1};
    assert(check_tensor(QTQ, I2, 4));

    tensor_destroy(A);
    tensor_destroy(Q);
    tensor_destroy(R);
    tensor_destroy(QR);
    tensor_destroy(QT);
    tensor_destroy(QTQ);
    PASS();
}

/* 秩亏矩阵的 SVD 分解测试 */
void test_svd_rank_deficient()
{
    TEST("tensor_svd rank deficient (multiple cases)");

    /* ---------- 测试用例1: 2x2 全1矩阵 (秩1) ---------- */
    {
        float data[] = {1, 1, 1, 1};
        Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
        Tensor *U = tensor_create(2, (int[]){2, 2});
        Tensor *S = tensor_create(1, (int[]){2});
        Tensor *V = tensor_create(2, (int[]){2, 2});
        tensor_svd(A, U, S, V, 1); // full

        // 重建 A = U * diag(S) * V^T
        Tensor *US = tensor_create(2, (int[]){2, 2});
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                US->data[i * 2 + j] = U->data[i * 2 + j] * S->data[j];
        Tensor *VT = tensor_create(2, (int[]){2, 2});
        tensor_permute(V, (int[]){1, 0}, VT);
        Tensor *A_recon = tensor_create(2, (int[]){2, 2});
        tensor_matmul(US, VT, A_recon);

        float local_eps = 1e-3f;
        int close = 1;
        for (int i = 0; i < 4; i++)
        {
            if (!approx_equal(A->data[i], A_recon->data[i], local_eps))
            {
                close = 0;
                break;
            }
        }
        assert(close);

        tensor_destroy(A);
        tensor_destroy(U);
        tensor_destroy(S);
        tensor_destroy(V);
        tensor_destroy(US);
        tensor_destroy(VT);
        tensor_destroy(A_recon);
    }

    /* ---------- 测试用例2: 3x3 矩阵 [1 2 3; 4 5 6; 7 8 9] (秩2) ---------- */
    {
        float data[] = {1, 2, 3,
                        4, 5, 6,
                        7, 8, 9};
        Tensor *A = tensor_from_array(data, 2, (int[]){3, 3});
        Tensor *U = tensor_create(2, (int[]){3, 3});
        Tensor *S = tensor_create(1, (int[]){3});
        Tensor *V = tensor_create(2, (int[]){3, 3});
        tensor_svd(A, U, S, V, 1); // full

        // 重建 A = U * diag(S) * V^T
        Tensor *US = tensor_create(2, (int[]){3, 3});
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                US->data[i * 3 + j] = U->data[i * 3 + j] * S->data[j];
        Tensor *VT = tensor_create(2, (int[]){3, 3});
        tensor_permute(V, (int[]){1, 0}, VT);
        Tensor *A_recon = tensor_create(2, (int[]){3, 3});
        tensor_matmul(US, VT, A_recon);

        float local_eps = 1e-3f;
        int close = 1;
        for (int i = 0; i < 9; i++)
        {
            if (!approx_equal(A->data[i], A_recon->data[i], local_eps))
            {
                close = 0;
                break;
            }
        }
        assert(close);

        // 验证秩2：第三个奇异值应接近0
        assert(fabs(S->data[2]) < 1e-2f);

        tensor_destroy(A);
        tensor_destroy(U);
        tensor_destroy(S);
        tensor_destroy(V);
        tensor_destroy(US);
        tensor_destroy(VT);
        tensor_destroy(A_recon);
    }

    PASS();
}

/* tensor_eigh 对非对称矩阵返回错误 */
void test_eigh_nonsymmetric()
{
    TEST("tensor_eigh nonsymmetric");
    float data[] = {1, 2, 3, 4}; // 非对称
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *eigvals = tensor_create(1, (int[]){2});
    Tensor *eigvecs = tensor_create(2, (int[]){2, 2});
    TensorStatus st = tensor_eigh(A, eigvals, eigvecs);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(A);
    tensor_destroy(eigvals);
    tensor_destroy(eigvecs);
    PASS();
}

/* tensor_tensordot 多轴缩并 */
void test_tensordot_multi_axis()
{
    TEST("tensor_tensordot multi-axis");
    // 使用全1张量，简化验证
    int dims_a[] = {2, 3, 4};
    int dims_b[] = {4, 3, 2};
    float *data_a = (float *)malloc(24 * sizeof(float));
    float *data_b = (float *)malloc(24 * sizeof(float));
    for (int i = 0; i < 24; i++)
    {
        data_a[i] = 1.0f;
        data_b[i] = 1.0f;
    }
    Tensor *a = tensor_from_array(data_a, 3, dims_a);
    Tensor *b = tensor_from_array(data_b, 3, dims_b);
    free(data_a);
    free(data_b);

    int axes_a[] = {1, 2};
    int axes_b[] = {1, 0};
    // 输出形状 (2,2)
    int out_dims[] = {2, 2};
    Tensor *out = tensor_create(2, out_dims);
    TensorStatus st = tensor_tensordot(a, b, axes_a, axes_b, 2, out);
    assert(st == TENSOR_OK);

    // 缩并元素总数为 3*4 = 12，每个乘积为 1，所以每个输出元素应为 12
    float expected[4] = {12.0f, 12.0f, 12.0f, 12.0f};
    assert(check_tensor(out, expected, 4));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* tensor_permute 无效 axes 测试 */
void test_permute_invalid_axes()
{
    TEST("tensor_permute invalid axes");
    int dims[] = {2, 3, 4};
    Tensor *a = tensor_create(3, dims);

    // 重复轴：axes = {0,1,1}，此时 permute 后的形状应为 [2,3,3]
    Tensor *out1 = tensor_create(3, (int[]){2, 3, 3});
    int axes_dup[] = {0, 1, 1};
    TensorStatus st = tensor_permute(a, axes_dup, out1);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(out1);

    // 越界轴：axes = {0,3,1}，其中 3 越界
    Tensor *out2 = tensor_create(3, (int[]){2, 4, 3}); // 形状任意，因为会先检查轴
    int axes_out_of_range[] = {0, 3, 1};
    st = tensor_permute(a, axes_out_of_range, out2);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(out2);

    tensor_destroy(a);
    PASS();
}

/* tensor_diag 输入非1D/2D 返回错误 */
void test_diag_invalid_ndim()
{
    TEST("tensor_diag invalid ndim");
    int dims[] = {2, 3, 4};
    Tensor *a = tensor_create(3, dims);
    Tensor *out = tensor_create(1, (int[]){3}); // 随便
    TensorStatus st = tensor_diag(a, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* tensor_transpose 输入小于2维返回错误 */
void test_transpose_invalid_ndim()
{
    TEST("tensor_transpose invalid ndim");
    Tensor *a = tensor_create(1, (int[]){5});
    Tensor *out = tensor_create(1, (int[]){5});
    TensorStatus st = tensor_transpose(a, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);
    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* tensor_trace 轴相同或越界 */
void test_trace_invalid_axes()
{
    TEST("tensor_trace invalid axes");
    int dims[] = {3, 3, 3};
    Tensor *a = tensor_create(3, dims);
    Tensor *out = tensor_create(1, (int[]){3});

    // 轴相同
    TensorStatus st = tensor_trace(a, 1, 1, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    // 轴越界
    st = tensor_trace(a, 1, 3, out);
    assert(st == TENSOR_ERR_INVALID_PARAM);

    tensor_destroy(a);
    tensor_destroy(out);
    PASS();
}

/* 不连续输入测试：对转置视图执行矩阵乘法，结果应与连续版本一致 */
void test_matmul_noncontiguous()
{
    TEST("tensor_matmul noncontiguous");
    // 创建连续矩阵 A_cont(2,3) 和 B_cont(3,2)
    float data_a[] = {1, 2, 3, 4, 5, 6};
    float data_b[] = {7, 8, 9, 10, 11, 12};
    Tensor *A_cont = tensor_from_array(data_a, 2, (int[]){2, 3});
    Tensor *B_cont = tensor_from_array(data_b, 2, (int[]){3, 2});

    // 创建不连续视图：A_view 形状 (2,3) 步长 [1,2]（列主序布局）
    int strides_A[] = {1, 2};
    Tensor *A_view = tensor_view(A_cont, 2, (int[]){2, 3}, strides_A);
    assert(A_view != NULL);
    // B_view 形状 (3,2) 步长 [1,3]
    int strides_B[] = {1, 3};
    Tensor *B_view = tensor_view(B_cont, 2, (int[]){3, 2}, strides_B);
    assert(B_view != NULL);

    // 预期结果：根据逻辑布局手动计算
    // A_view 逻辑矩阵：
    // [ [1, 3, 5],
    //   [2, 4, 6] ]
    // B_view 逻辑矩阵：
    // [ [7, 10],
    //   [8, 11],
    //   [9, 12] ]
    float expected[] = {
        1 * 7 + 3 * 8 + 5 * 9, 1 * 10 + 3 * 11 + 5 * 12,
        2 * 7 + 4 * 8 + 6 * 9, 2 * 10 + 4 * 11 + 6 * 12}; // 预期 [76, 103, 100, 136]

    Tensor *C_view = tensor_create(2, (int[]){2, 2});
    tensor_matmul(A_view, B_view, C_view);

    assert(check_tensor(C_view, expected, 4));

    tensor_destroy(A_cont);
    tensor_destroy(B_cont);
    tensor_destroy(A_view);
    tensor_destroy(B_view);
    tensor_destroy(C_view);
    PASS();
}

/* 空指针测试 */
void test_null_ptr()
{
    TEST("tensor_matmul null ptr");
    Tensor *a = tensor_create(2, (int[]){2, 2});
    Tensor *b = tensor_create(2, (int[]){2, 2});
    Tensor *out = tensor_create(2, (int[]){2, 2});
    TensorStatus st = tensor_matmul(NULL, b, out);
    assert(st == TENSOR_ERR_NULL_PTR);
    st = tensor_matmul(a, NULL, out);
    assert(st == TENSOR_ERR_NULL_PTR);
    st = tensor_matmul(a, b, NULL);
    assert(st == TENSOR_ERR_NULL_PTR);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

/* 形状不匹配测试 */
void test_shape_mismatch()
{
    TEST("tensor_matmul shape mismatch");
    Tensor *a = tensor_create(2, (int[]){2, 3});
    Tensor *b = tensor_create(2, (int[]){4, 5}); // 内维不匹配
    Tensor *out = tensor_create(2, (int[]){2, 5});
    TensorStatus st = tensor_matmul(a, b, out);
    assert(st == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(out);
    PASS();
}

void test_eig_2x2_real()
{
    TEST("tensor_eig 2x2 real");
    // 矩阵 [2, 1; 1, 2] 特征值 3 和 1
    float data[] = {2, 1, 1, 2};
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *eigr = tensor_create(1, (int[]){2});
    Tensor *eigi = tensor_create(1, (int[]){2});
    Tensor *vecr = tensor_create(2, (int[]){2, 2});
    Tensor *veci = tensor_create(2, (int[]){2, 2});

    TensorStatus st = tensor_eig(A, eigr, eigi, vecr, veci);
    assert(st == TENSOR_OK);

    float expected_r[] = {3, 1};
    float expected_i[] = {0, 0};
    assert(check_tensor(eigr, expected_r, 2));
    assert(check_tensor(eigi, expected_i, 2));

    // 验证特征向量正交（近似）
    float dot = vecr->data[0*2+0]*vecr->data[1*2+0] + vecr->data[0*2+1]*vecr->data[1*2+1];
    assert(fabs(dot) < EPS);

    tensor_destroy(A);
    tensor_destroy(eigr);
    tensor_destroy(eigi);
    tensor_destroy(vecr);
    tensor_destroy(veci);
    PASS();
}

void test_eig_2x2_complex()
{
    TEST("tensor_eig 2x2 complex");
    // 旋转矩阵 [0, -1; 1, 0] 特征值 ±i
    float data[] = {0, -1, 1, 0};
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *eigr = tensor_create(1, (int[]){2});
    Tensor *eigi = tensor_create(1, (int[]){2});
    Tensor *vecr = tensor_create(2, (int[]){2, 2});
    Tensor *veci = tensor_create(2, (int[]){2, 2});

    TensorStatus st = tensor_eig(A, eigr, eigi, vecr, veci);
    assert(st == TENSOR_OK); // 期望成功

    // 特征值应为 0±i
    float expected_r[] = {0, 0};
    float expected_i[] = {1, -1};
    assert(check_tensor(eigr, expected_r, 2));
    assert(check_tensor(eigi, expected_i, 2));

    // 特征向量实部/虚部检查（略）
    tensor_destroy(A);
    tensor_destroy(eigr);
    tensor_destroy(eigi);
    tensor_destroy(vecr);
    tensor_destroy(veci);
    PASS();
}

void test_lstsq_overdetermined()
{
    TEST("tensor_lstsq overdetermined");
    // A: 3x2, B: 3x1
    float A_data[] = {1, 1, 2, 1, 3, 1}; // 列：x, 1
    float B_data[] = {2, 3, 4};
    Tensor *A = tensor_from_array(A_data, 2, (int[]){3, 2});
    Tensor *B = tensor_from_array(B_data, 1, (int[]){3});
    Tensor *X = tensor_create(1, (int[]){2});

    tensor_lstsq(A, B, X);
    // 最小二乘解应为 [1, 1] (因为 y = x + 1)
    float expected[] = {1, 1};
    assert(check_tensor(X, expected, 2));

    tensor_destroy(A);
    tensor_destroy(B);
    tensor_destroy(X);
    PASS();
}

void test_lstsq_underdetermined()
{
    TEST("tensor_lstsq underdetermined");
    // A: 2x3, B: 2x1, 最小范数解
    float A_data[] = {1, 0, 0, 0, 1, 0};
    float B_data[] = {2, 3};
    Tensor *A = tensor_from_array(A_data, 2, (int[]){2, 3});
    Tensor *B = tensor_from_array(B_data, 1, (int[]){2});
    Tensor *X = tensor_create(1, (int[]){3});

    tensor_lstsq(A, B, X);
    float expected[] = {2, 3, 0};
    assert(check_tensor(X, expected, 3));

    tensor_destroy(A);
    tensor_destroy(B);
    tensor_destroy(X);
    PASS();
}

void test_lstsq_rank_deficient()
{
    TEST("tensor_lstsq rank deficient");
    // A: 3x2 秩1
    float A_data[] = {1, 2, 2, 4, 3, 6};
    float B_data[] = {3, 6, 9};
    Tensor *A = tensor_from_array(A_data, 2, (int[]){3, 2});
    Tensor *B = tensor_from_array(B_data, 1, (int[]){3});
    Tensor *X = tensor_create(1, (int[]){2});

    tensor_lstsq(A, B, X);
    // 最小范数解应为 [1, 2] 的倍数，实际解应为 [1, 2] 左右（因为 B 正比于 A 的第一列）
    // 具体值可能因算法而异，只需验证残差较小
    Tensor *AX = tensor_create(1, (int[]){3});
    tensor_matmul(A, X, AX);
    float diff = 0;
    for (int i = 0; i < 3; i++)
        diff += fabs(AX->data[i] - B_data[i]);
    assert(diff < EPS * 3);

    tensor_destroy(A);
    tensor_destroy(B);
    tensor_destroy(X);
    tensor_destroy(AX);
    PASS();
}

void test_matrix_rank_full()
{
    TEST("tensor_matrix_rank full");
    float data[] = {1, 2, 3, 4};
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *out = tensor_create(0, NULL);
    tensor_matrix_rank(A, 0, out);
    assert(out->data[0] == 2);
    tensor_destroy(A);
    tensor_destroy(out);
    PASS();
}

void test_matrix_rank_deficient()
{
    TEST("tensor_matrix_rank deficient");
    float data[] = {1, 2, 2, 4};
    Tensor *A = tensor_from_array(data, 2, (int[]){2, 2});
    Tensor *out = tensor_create(0, NULL);
    tensor_matrix_rank(A, 1e-5, out);
    assert(out->data[0] == 1);
    tensor_destroy(A);
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
    test_det_2x2();
    test_det_3x3();
    test_solve_2x2();
    test_solve_3x3();
    test_cholesky_3x3();
    test_svd_3x2_reduced();
    test_svd_3x2_full();
    test_qr_3x2_reduced();
    test_qr_3x2_full();
    test_eigh_3x3();

    // 新增错误测试
    test_inv_singular();
    test_det_singular();
    test_solve_singular();
    test_cholesky_non_positive_definite();
    test_qr_rank_deficient();
    test_svd_rank_deficient();
    test_eigh_nonsymmetric();
    test_tensordot_multi_axis();
    test_permute_invalid_axes();
    test_diag_invalid_ndim();
    test_transpose_invalid_ndim();
    test_trace_invalid_axes();
    test_matmul_noncontiguous();
    test_null_ptr();
    test_shape_mismatch();

 test_eig_2x2_real();
    test_eig_2x2_complex();
    test_lstsq_overdetermined();
    test_lstsq_underdetermined();
    test_lstsq_rank_deficient();
    test_matrix_rank_full();
    test_matrix_rank_deficient();

    printf("All linalg_ops tests passed!\n");
    return 0;
}