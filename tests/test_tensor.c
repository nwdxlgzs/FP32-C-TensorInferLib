#include "tensor.h"
#include "utils.h"
#include "shape_ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

/* 辅助：打印测试结果 */
#define TEST(name) printf("Running %s ... ", name)
#define PASS() printf("PASSED\n")

/* 辅助：浮点数近似比较（如果数值不是精确整数时使用） */
#define ASSERT_FLOAT_EQ(a, b)          \
    do                                 \
    {                                  \
        float diff = fabsf((a) - (b)); \
        assert(diff < 1e-5f);          \
    } while (0)

/* ==================== 原有测试（保持不变） ==================== */

void test_create_destroy()
{
    TEST("test_create_destroy");
    int dims[] = {2, 3};
    Tensor *t = tensor_create(2, dims);
    assert(t != NULL);
    assert(tensor_ndim(t) == 2);
    assert(tensor_dims(t)[0] == 2 && tensor_dims(t)[1] == 3);
    assert(tensor_size(t) == 6);
    assert(tensor_strides(t) != NULL);
    assert(tensor_strides(t)[0] == 3 && tensor_strides(t)[1] == 1);
    tensor_destroy(t);
    PASS();
}

void test_create_scalar()
{
    TEST("test_create_scalar");
    Tensor *t = tensor_create(0, NULL);
    assert(t != NULL);
    assert(tensor_ndim(t) == 0);
    assert(tensor_dims(t) == NULL);
    assert(tensor_strides(t) == NULL);
    assert(tensor_size(t) == 1);
    tensor_destroy(t);
    PASS();
}

void test_wrap()
{
    TEST("test_wrap");
    float data[6] = {1, 2, 3, 4, 5, 6};
    int dims[] = {2, 3};
    Tensor *t = tensor_wrap(data, 2, dims, NULL);
    assert(t != NULL);
    assert(tensor_ndim(t) == 2);
    assert(tensor_size(t) == 6);
    assert(tensor_strides(t) == NULL); // 标记连续
    ptrdiff_t off = tensor_offset(t, (int[]){1, 2});
    assert(off == 1 * 3 + 2);
    assert(t->data[off] == 6);
    tensor_destroy(t); // 不会释放data
    PASS();
}

void test_from_array()
{
    TEST("test_from_array");
    float data[6] = {1, 2, 3, 4, 5, 6};
    int dims[] = {2, 3};
    Tensor *t = tensor_from_array(data, 2, dims);
    assert(t != NULL);
    assert(t->data[0] == 1);
    data[0] = 100; // 修改原数组不影响t
    assert(t->data[0] == 1);
    tensor_destroy(t);
    PASS();
}

void test_clone()
{
    TEST("test_clone");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 4; ++i)
        a->data[i] = i;
    Tensor *b = tensor_clone(a);
    assert(b != NULL);
    assert(b->data[0] == 0 && b->data[3] == 3);
    a->data[0] = 100;
    assert(b->data[0] == 0);
    tensor_destroy(a);
    tensor_destroy(b);
    PASS();
}

void test_view_reshape()
{
    TEST("test_view_reshape");
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    // 创建视图 (3,2) 通过 reshape（原连续，所以自动计算连续步长）
    int new_dims[] = {3, 2};
    Tensor *v = tensor_view(a, 2, new_dims, NULL);
    assert(v != NULL);
    assert(v->strides[0] == 2 && v->strides[1] == 1); // 自动计算的连续步长
    // 验证数据对应关系
    assert(v->data[0] == a->data[0]); // v(0,0) -> a(0,0)
    int idx_v[2] = {2, 1};            // 对应 a 的哪个？a 连续步长(3,1)，v(2,1) 线性索引 2*2+1=5，对应 a 的线性索引 5，即 a(1,2) 值为 5
    ptrdiff_t off_v = tensor_offset(v, idx_v);
    assert(off_v == 5);
    assert(v->data[off_v] == 5);
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_view_with_strides()
{
    TEST("test_view_with_strides");
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    // 创建转置视图：形状 (3,2)，步长 (1,3) 使得 v(i,j) = a(j,i)
    int new_dims[] = {3, 2};
    int strides[] = {1, 3};
    Tensor *v = tensor_view(a, 2, new_dims, strides);
    assert(v != NULL);
    assert(v->strides[0] == 1 && v->strides[1] == 3);
    // 验证转置：
    // v(0,0) = a(0,0) = 0
    // v(0,1) = a(1,0) = 3
    // v(1,0) = a(0,1) = 1
    // v(2,1) = a(1,2) = 5
    assert(v->data[tensor_offset(v, (int[]){0, 0})] == 0);
    assert(v->data[tensor_offset(v, (int[]){0, 1})] == 3);
    assert(v->data[tensor_offset(v, (int[]){1, 0})] == 1);
    assert(v->data[tensor_offset(v, (int[]){2, 1})] == 5);
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_view_noncontiguous_fail()
{
    TEST("test_view_noncontiguous_fail");
    // 创建一个不连续的张量（通过视图转置）
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    int new_dims[] = {3, 2};
    int strides[] = {1, 3};
    Tensor *v = tensor_view(a, 2, new_dims, strides); // v 不连续
    assert(v != NULL);
    // 尝试在不连续的原张量上创建视图而不提供步长，应该失败
    int new_dims2[] = {1, 6};
    Tensor *w = tensor_view(v, 2, new_dims2, NULL);
    assert(w == NULL); // 应该失败
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_copy()
{
    TEST("test_copy");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 4; ++i)
        a->data[i] = i;
    Tensor *b = tensor_create(2, dims);
    TensorStatus status = tensor_copy(b, a);
    assert(status == TENSOR_OK);
    assert(b->data[0] == 0 && b->data[3] == 3);
    a->data[0] = 100;
    assert(b->data[0] == 0);
    tensor_destroy(a);
    tensor_destroy(b);
    PASS();
}

void test_contiguous()
{
    TEST("test_contiguous");
    int dims[] = {3, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    // 创建转置视图 v: 形状 (2,3)，步长 (1,2) 使得 v(i,j) = a(j,i)
    int new_dims[] = {2, 3};
    int strides[] = {1, 2}; // 修正：正确的转置步长
    Tensor *v = tensor_view(a, 2, new_dims, strides);
    assert(!util_is_contiguous(v)); // 不连续
    // 连续化
    TensorStatus status = tensor_contiguous(v);
    assert(status == TENSOR_OK);
    assert(util_is_contiguous(v));
    // 验证数据：转置后的连续数据应为 [0,2,4,1,3,5]
    float expected[] = {0, 2, 4, 1, 3, 5};
    for (int i = 0; i < 6; ++i)
        assert(v->data[i] == expected[i]);
    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

void test_contiguous_external_fail()
{
    TEST("test_contiguous_external_fail");
    float data[6] = {0, 1, 2, 3, 4, 5};
    int dims[] = {3, 2};
    // 包装为连续张量
    Tensor *t = tensor_wrap(data, 2, dims, NULL);
    // 制造一个不连续的外部视图？但 wrap 只能包装连续数据，除非提供非连续步长
    // 这里测试外部数据且不连续的情况：提供非连续步长
    int strides[] = {1, 3};
    Tensor *v = tensor_wrap(data, 2, dims, strides); // 外部数据，不连续
    assert(v != NULL);
    assert(!util_is_contiguous(v));
    // 连续化应该失败，因为外部数据无法原地变连续
    TensorStatus status = tensor_contiguous(v);
    assert(status == TENSOR_ERR_UNSUPPORTED);
    tensor_destroy(v);
    // 用 wrap 连续数据则没问题
    Tensor *w = tensor_wrap(data, 2, dims, NULL);
    status = tensor_contiguous(w);
    assert(status == TENSOR_OK); // 已经是连续
    tensor_destroy(w);
    PASS();
}

void test_offset()
{
    TEST("test_offset");
    int dims[] = {2, 3, 4};
    Tensor *t = tensor_create(3, dims);
    ptrdiff_t off = tensor_offset(t, (int[]){1, 2, 3});
    assert(off == 1 * 12 + 2 * 4 + 3); // 12+8+3=23
    // 负索引
    off = tensor_offset(t, (int[]){-1, -1, -1});
    assert(off == 1 * 12 + 2 * 4 + 3);
    // 越界
    off = tensor_offset(t, (int[]){2, 0, 0});
    assert(off == SIZE_MAX);
    tensor_destroy(t);
    PASS();
}

void test_make_unique()
{
    TEST("test_make_unique");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 4; ++i)
        a->data[i] = i;
    Tensor *b = tensor_view(a, 2, dims, NULL); // 共享数据
    assert(*(a->ref_count) == 2);
    // 通过 b 触发写时拷贝（模拟修改）
    TensorStatus status = tensor_make_unique(b);
    assert(status == TENSOR_OK);
    assert(*(a->ref_count) == 1);
    assert(*(b->ref_count) == 1);
    assert(a->data != b->data);
    // 修改 b 不影响 a
    b->data[0] = 100;
    assert(a->data[0] == 0);
    tensor_destroy(a);
    tensor_destroy(b);
    PASS();
}

/* ==================== 新增测试 ==================== */

/* 测试 tensor_cleanup 正确释放资源并递减引用计数 */
void test_cleanup()
{
    TEST("test_cleanup");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    a->data[0] = 1;
    a->data[1] = 2;
    a->data[2] = 3;
    a->data[3] = 4;
    Tensor *b = tensor_view(a, 2, dims, NULL); // 共享
    assert(*(a->ref_count) == 2);

    tensor_cleanup(b); // 清理 b，应递减引用计数
    assert(*(a->ref_count) == 1);
    assert(b->data == NULL && b->dims == NULL && b->strides == NULL && b->ref_count == NULL);

    tensor_destroy(a); // 最终释放
    PASS();
}

/* 测试 tensor_dim_size 函数 */
void test_dim_size()
{
    TEST("test_dim_size");
    int dims[] = {2, 3, 4};
    Tensor *t = tensor_create(3, dims);

    assert(tensor_dim_size(t, 0) == 2);
    assert(tensor_dim_size(t, 1) == 3);
    assert(tensor_dim_size(t, 2) == 4);
    assert(tensor_dim_size(t, -1) == 4); // 负索引
    assert(tensor_dim_size(t, -2) == 3);
    assert(tensor_dim_size(t, -3) == 2);
    assert(tensor_dim_size(t, 3) == -1);  // 越界
    assert(tensor_dim_size(t, -4) == -1); // 越界

    tensor_destroy(t);
    PASS();
}

/* 测试多个视图下写时拷贝的正确性 */
void test_make_unique_multiple_views()
{
    TEST("test_make_unique_multiple_views");
    int dims[] = {2, 2};
    Tensor *a = tensor_create(2, dims);
    a->data[0] = 1;
    a->data[1] = 2;
    a->data[2] = 3;
    a->data[3] = 4;
    Tensor *v1 = tensor_view(a, 2, dims, NULL);
    Tensor *v2 = tensor_view(a, 2, dims, NULL);
    assert(*(a->ref_count) == 3);

    tensor_make_unique(v1); // 触发写时拷贝
    assert(*(a->ref_count) == 2);
    assert(*(v1->ref_count) == 1);
    assert(a->data != v1->data);

    v1->data[0] = 100; // 修改 v1 不影响 a
    assert(a->data[0] == 1);
    assert(v2->data[0] == 1); // v2 仍共享 a

    tensor_destroy(a);
    tensor_destroy(v1);
    tensor_destroy(v2);
    PASS();
}

/* 测试在不连续张量上调用 tensor_reshape 应返回错误 */
void test_reshape_noncontiguous()
{
    TEST("test_reshape_noncontiguous");
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    for (int i = 0; i < 6; ++i)
        a->data[i] = i;
    // 创建转置视图使之不连续
    int trans_dims[] = {3, 2};
    int strides[] = {1, 3};
    Tensor *v = tensor_view(a, 2, trans_dims, strides);
    assert(!util_is_contiguous(v));

    TensorStatus status = tensor_reshape(v, 1, (int[]){6});
    assert(status == TENSOR_ERR_UNSUPPORTED);

    tensor_destroy(v);
    tensor_destroy(a);
    PASS();
}

/* 测试 tensor_create 非法参数返回 NULL */
void test_create_invalid()
{
    TEST("test_create_invalid");
    // ndim < 0
    assert(tensor_create(-1, NULL) == NULL);
    // ndim > 0 但 dims 为 NULL
    assert(tensor_create(2, NULL) == NULL);
    // 维度包含0（创建空张量？目前库禁止空张量）
    int dims0[] = {2, 0};
    assert(tensor_create(2, dims0) == NULL); // 因为 size 计算为0
    PASS();
}

/* 测试 tensor_wrap 非法参数返回 NULL */
void test_wrap_invalid()
{
    TEST("test_wrap_invalid");
    float dummy = 0;
    // data 为 NULL
    assert(tensor_wrap(NULL, 1, (int[]){2}, NULL) == NULL);
    // ndim < 0
    assert(tensor_wrap(&dummy, -1, NULL, NULL) == NULL);
    // ndim > 0 且 dims 为 NULL
    assert(tensor_wrap(&dummy, 1, NULL, NULL) == NULL);
    // 包含零维度（size 为0）
    assert(tensor_wrap(&dummy, 2, (int[]){2, 0}, NULL) == NULL);
    PASS();
}

/* 测试 tensor_offset 越界返回 SIZE_MAX 的更多情况 */
void test_offset_out_of_bounds()
{
    TEST("test_offset_out_of_bounds");
    int dims[] = {2, 3};
    Tensor *t = tensor_create(2, dims);
    // 单个坐标越界
    assert(tensor_offset(t, (int[]){2, 0}) == SIZE_MAX);
    assert(tensor_offset(t, (int[]){0, 3}) == SIZE_MAX);
    // 多个越界
    assert(tensor_offset(t, (int[]){2, 3}) == SIZE_MAX);
    // 负索引修正后越界
    assert(tensor_offset(t, (int[]){-3, 0}) == SIZE_MAX); // -3+2 = -1
    assert(tensor_offset(t, (int[]){0, -4}) == SIZE_MAX); // -4+3 = -1
    tensor_destroy(t);
    PASS();
}

/* 测试 tensor_copy 形状不匹配返回错误 */
void test_copy_shape_mismatch()
{
    TEST("test_copy_shape_mismatch");
    int dims1[] = {2, 2};
    int dims2[] = {2, 3};
    Tensor *a = tensor_create(2, dims1);
    Tensor *b = tensor_create(2, dims2);
    TensorStatus status = tensor_copy(b, a);
    assert(status == TENSOR_ERR_SHAPE_MISMATCH);
    tensor_destroy(a);
    tensor_destroy(b);
    PASS();
}

/* 测试 tensor_view 因总数不匹配而失败 */
void test_view_fail_total()
{
    TEST("test_view_fail_total");
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    // 新形状元素总数不等于原张量
    int bad_dims[] = {3, 3}; // 9 != 6
    Tensor *v = tensor_view(a, 2, bad_dims, NULL);
    assert(v == NULL);
    tensor_destroy(a);
    PASS();
}

/* 测试 tensor_view 在不连续且无步长时失败 */
void test_view_fail_noncontiguous()
{
    TEST("test_view_fail_noncontiguous");
    int dims[] = {2, 3};
    Tensor *a = tensor_create(2, dims);
    // 创建不连续视图（转置）
    int trans_dims[] = {3, 2};
    int strides[] = {1, 3};
    Tensor *v1 = tensor_view(a, 2, trans_dims, strides);
    assert(v1 != NULL);
    // 试图在不连续的 v1 上创建新视图而不提供步长
    int new_dims[] = {1, 6};
    Tensor *v2 = tensor_view(v1, 2, new_dims, NULL);
    assert(v2 == NULL);
    tensor_destroy(v1);
    tensor_destroy(a);
    PASS();
}

/* ==================== main ==================== */
int main()
{
    test_create_destroy();
    test_create_scalar();
    test_wrap();
    test_from_array();
    test_clone();
    test_view_reshape();
    test_view_with_strides();
    test_view_noncontiguous_fail();
    test_copy();
    test_contiguous();
    test_contiguous_external_fail();
    test_offset();
    test_make_unique();

    // 新增测试
    test_cleanup();
    test_dim_size();
    test_make_unique_multiple_views();
    test_reshape_noncontiguous();
    test_create_invalid();
    test_wrap_invalid();
    test_offset_out_of_bounds();
    test_copy_shape_mismatch();
    test_view_fail_total();
    test_view_fail_noncontiguous();

    printf("All tensor tests passed!\n");
    return 0;
}