# FP32-C-TensorInferLib

**Platform‑independent FP32 tensor library for inference written in pure C**  
**纯C编写的、平台无关的FP32推理张量库**

*No external dependencies – just standard C (C99/C11)*  
*无外部依赖——仅需标准C（C99/C11）*

FP32‑C‑TensorInferLib is a lightweight, stand‑alone tensor library designed for inference workloads.  
FP32‑C‑TensorInferLib 是一个为推理任务设计的轻量级、独立张量库。

It provides a comprehensive set of tensor operations – from basic arithmetic and reductions to linear algebra,
neural network layers, FFTs, and advanced indexing – all operating on **single‑precision (float)** data.  
它提供了一整套张量操作——从基本算术、归约到线性代数、神经网络层、FFT以及高级索引——所有操作均基于**单精度（float）**数据。

The library is built around a core tensor structure with **reference counting**, **copy‑on‑write**, and **full broadcasting**
support. All operations work on both contiguous and strided (view‑based) tensors without unnecessary data duplication.  
该库围绕核心张量结构构建，支持**引用计数**、**写时拷贝**和**完整广播**。所有操作均适用于连续和带步长（视图）的张量，避免不必要的数据复制。

---

## Features
## 特性

- **Core tensor mechanics**  
- **核心张量机制**  
  - Multi‑dimensional arrays (up to `TENSOR_MAX_DIM`, default 8)  
  - 多维数组（最大维度 `TENSOR_MAX_DIM`，默认为8）  
  - Reference‑counted data with copy‑on‑write (`tensor_make_unique`)  
  - 引用计数数据，支持写时拷贝（`tensor_make_unique`）  
  - Views (slices, transposes, reshapes) share the same data buffer  
  - 视图（切片、转置、变形）共享同一数据缓冲区  
  - Row‑major memory layout with explicit strides – supports arbitrary non‑contiguous views  
  - 行主序内存布局，显式步长——支持任意非连续视图  

- **Broadcasting**  
- **广播**  
  - NumPy‑style broadcasting for all binary/ternary operations  
  - 所有二元/三元运算均支持NumPy风格的广播  
  - Automatic broadcast‑shape computation and efficient iteration  
  - 自动计算广播形状并高效迭代  

- **Operations** (grouped by header)  
- **操作**（按头文件分组）  

  | Category               | Headers / Files                                  | Examples                                   |
  |------------------------|--------------------------------------------------|--------------------------------------------|
  | **Math**               | `math_ops.h`                                     | `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log`, `sin`, `cos`, `erf`, `sigmoid`, … |
  | **Compare & Logic**    | `compare_ops.h`                                  | `equal`, `less`, `greater`, `and`, `or`, `not` (return 0.0/1.0) |
  | **Reduce**             | `reduce_ops.h`                                   | `sum`, `mean`, `max`, `min`, `argmax`, `var`, `std`, `norm`, `median`, `mode`, `quantile` |
  | **Search & Sort**      | `search_ops.h`                                   | `sort`, `argsort`, `unique`, `topk`, `kthvalue`, `searchsorted` |
  | **Linear Algebra**     | `linalg_ops.h`                                   | `matmul` (batched), `bmm`, `dot`, `tensordot`, `inv`, `det`, `solve`, `cholesky`, `qr`, `svd`, `eigh`, `lstsq`, `matrix_rank` |
  | **Neural Network**     | `nn_ops.h`                                       | `conv1d`/`2d`/`3d`, `conv_transpose`, `pool`, `batchnorm`, `layernorm`, `instance_norm`, `group_norm`, `lrn`, `linear`, `dropout`, `softmax`, `upsample`, `embedding` |
  | **Indexing**           | `indexing.h`                                     | `get_item`, `set_item`, `slice`, `advanced_index`, `masked_select`, `gather`/`scatter`, `take`/`put`, `nonzero` |
  | **Shape Manipulation** | `shape_ops.h`                                    | `reshape`, `flatten`, `squeeze`, `unsqueeze`, `concat`, `stack`, `split`, `repeat`, `tile`, `transpose`, `flip`, `pad`, `broadcast_to`, `roll`, `movedim` |
  | **Random**             | `random_ops.h`                                   | `seed`, `uniform`, `normal`, `truncated_normal`, `bernoulli`, `randint`, `shuffle` |
  | **FFT**                | `fft_ops.h`                                      | `rfft`, `irfft`, `fft`, `ifft` (supports arbitrary length, falls back to DFT for non‑power‑of‑two) |
  | **Utilities**          | `utils.h`                                        | printing, save/load, allclose, nan/inf checks, init helpers, broadcasting helpers, generic iterators |

- **Write‑once, read‑many philosophy** – copy‑on‑write ensures that modifying a view does not affect other views unless explicitly needed.  
- **一次写入、多次读取理念** – 写时拷贝确保修改一个视图不会影响其他视图，除非显式需要。

- **Thoroughly tested** – each module comes with a comprehensive test suite (see `test_*.c`).  
- **全面测试** – 每个模块都配有完整的测试套件（参见 `test_*.c`）。

---

## Build
## 构建

The library consists of a single **include directory** and a **source directory**.  
该库仅包含一个**头文件目录**和一个**源文件目录**。

No external libraries are required – just a C compiler (GCC, Clang, MSVC, etc.) and the standard math library `-lm`.  
无需外部库——只需一个C编译器（GCC、Clang、MSVC等）和标准数学库 `-lm`。

### Example (GCC)
### 示例（GCC）

```bash
# Compile all source files into an object archive (or directly into your project)
# 将所有源文件编译为对象存档（或直接编译进你的项目）
gcc -c -Iinclude src/*.c -lm
ar rcs libtensor.a *.o

# Or compile a test program directly
# 或者直接编译一个测试程序
gcc -Iinclude -o test_math test_math_ops.c src/*.c -lm
```

For convenience, Windows batch scripts (`runbuild_test_*.bat`) are provided to compile and run each test individually.  
为方便起见，提供了Windows批处理脚本（`runbuild_test_*.bat`），用于单独编译和运行每个测试。

---

## Quick Start
## 快速开始

Here is a minimal example that creates two tensors, adds them, and prints the result:  
以下是一个最小示例，创建两个张量、相加并打印结果：

```c
#include "tensor.h"
#include "math_ops.h"
#include "utils.h"

int main() {
    // Create a 2x3 tensor filled with 1..6
    // 创建一个2x3张量，填充1..6
    int dims[] = {2, 3};
    Tensor *a = tensor_from_array((float[]){1,2,3,4,5,6}, 2, dims);
    
    // Create a 3-element vector (will be broadcast to 2x3)
    // 创建一个3元素向量（将被广播为2x3）
    int dims_b[] = {3};
    Tensor *b = tensor_from_array((float[]){10,20,30}, 1, dims_b);
    
    // Allocate output tensor with the broadcasted shape (2x3)
    // 分配与广播后形状（2x3）一致的输出张量
    Tensor *c = tensor_create(2, dims);
    
    // c = a + b
    tensor_add(a, b, c);
    
    // Print the result
    // 打印结果
    tensor_print(c, "a + b", -1);
    
    // Clean up
    // 清理
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(c);
    return 0;
}
```

Output:  
输出：
```
a + b: ndim=2, size=6, [2,3]
[11 22 33 14 25 36]
```

---

## Core Concepts
## 核心概念

### 1. Tensor Structure
### 1. 张量结构
```c
typedef struct {
    float *data;       // data pointer                         // 数据指针
    int *dims;         // shape (length ndim)                  // 形状（长度 ndim）
    int *strides;      // strides (bytes per element? no – strides in elements) // 步长（以元素个数计，不是字节）
    int ndim;          // number of dimensions (0 = scalar)    // 维度数量（0表示标量）
    size_t size;       // total number of elements             // 总元素个数
    int *ref_count;    // reference counter (NULL if external data) // 引用计数器（外部数据时为NULL）
    int owns_dims_strides; // whether dims/strides are owned by this tensor // 该张量是否拥有dims和strides数组
} Tensor;
```

- `strides` are given in **number of elements**, not bytes.  
- `strides` 以**元素个数**为单位，不是字节。
- If `strides == NULL`, the tensor is assumed to be **row‑major contiguous** (computed on the fly).  
- 如果 `strides == NULL`，则假定张量为**行主序连续**（即时计算）。
- `ref_count` manages shared data; views increment the counter.  
- `ref_count` 管理共享数据；视图会增加计数器。
- `tensor_make_unique()` ensures the tensor has exclusive ownership of its data (copy‑on‑write).  
- `tensor_make_unique()` 确保张量独占其数据（写时拷贝）。

### 2. Broadcasting
### 2. 广播
All binary/ternary operations (e.g. `tensor_add`, `tensor_where`) follow the standard NumPy broadcasting rules:
- Two dimensions are compatible if they are equal or one of them is 1.
- The result shape is the element‑wise maximum of the two input shapes.
- The implementation efficiently iterates over the broadcasted shape without expanding memory.

所有二元/三元运算（例如 `tensor_add`、`tensor_where`）均遵循标准NumPy广播规则：
- 两个维度如果相等或其中之一为1，则兼容。
- 结果形状是输入形状逐元素取最大值。
- 实现高效地遍历广播后形状，不展开内存。

### 3. Views and Copy‑on‑Write
### 3. 视图与写时拷贝
Functions like `tensor_slice`, `tensor_transpose`, `tensor_reshape_view` return a new tensor that **shares** the data of the original.  
像 `tensor_slice`、`tensor_transpose`、`tensor_reshape_view` 这样的函数返回一个新张量，它**共享**原始张量的数据。

Modifying a view (e.g. via `tensor_set_item` or an in‑place operation) will first trigger a copy if the data is shared, ensuring isolation.  
修改一个视图（例如通过 `tensor_set_item` 或原地操作）如果数据被共享，则会先触发一次拷贝，确保隔离。

### 4. Error Handling
### 4. 错误处理
All public functions return a `TensorStatus` enum. Success is `TENSOR_OK`.  
所有公共函数返回 `TensorStatus` 枚举。成功为 `TENSOR_OK`。

Use `tensor_status_to_string(status)` to get a human‑readable error message.  
使用 `tensor_status_to_string(status)` 获取人类可读的错误信息。

---

## API Overview
## API 概览

The API is split into several headers. For detailed documentation of each function, please refer to the corresponding `.h` file.  
API 分为多个头文件。每个函数的详细文档请参考对应的 `.h` 文件。

| Header          | Purpose                                                                 | 用途                                                       |
|-----------------|-------------------------------------------------------------------------|------------------------------------------------------------|
| `tensor.h`      | Core tensor creation, destruction, copying, views, attributes.         | 核心张量创建、销毁、复制、视图、属性。                     |
| `math_ops.h`    | Elementary math functions (unary, binary, ternary, special functions). | 基础数学函数（一元、二元、三元、特殊函数）。                 |
| `compare_ops.h` | Comparison (`==`, `<`, `>`, …) and logical (`&&`, `||`, `!`).          | 比较（`==`、`<`、`>`等）和逻辑（`&&`、`||`、`!`）运算。     |
| `reduce_ops.h`  | Reductions (sum, mean, max, argmax, variance, norm, median, …).        | 归约（求和、均值、最大值、argmax、方差、范数、中位数等）。 |
| `search_ops.h`  | Sorting, searching, `topk`, `kthvalue`, `unique`, `searchsorted`.      | 排序、搜索、`topk`、`kthvalue`、`unique`、`searchsorted`。 |
| `linalg_ops.h`  | Matrix multiplication, decompositions (LU, Cholesky, QR, SVD, EVD), linear solvers, least squares. | 矩阵乘法、分解（LU、Cholesky、QR、SVD、EVD）、线性求解器、最小二乘。 |
| `nn_ops.h`      | Neural network layers (convolution, pooling, normalization, activation, dropout, embedding, upsampling). | 神经网络层（卷积、池化、归一化、激活、dropout、嵌入、上采样）。 |
| `indexing.h`    | Single element access, slicing, advanced integer / boolean indexing, `gather`/`scatter`, `take`/`put`. | 单元素访问、切片、高级整数/布尔索引、`gather`/`scatter`、`take`/`put`。 |
| `shape_ops.h`   | Reshaping, concatenation, stacking, splitting, repeating, tiling, transposition, padding, rolling, moving axes. | 变形、拼接、堆叠、分割、重复、平铺、转置、填充、滚动、移动轴。 |
| `random_ops.h`  | Random number generation (uniform, normal, Bernoulli, shuffling).      | 随机数生成（均匀、正态、伯努利、打乱）。                     |
| `fft_ops.h`     | Real and complex FFT (radix‑2 for powers of two, fallback DFT otherwise). | 实数和复数FFT（2的幂使用基2算法，否则回退DFT）。           |
| `utils.h`       | Debug printing, file I/O, `allclose`, nan/inf checks, broadcasting helpers, generic iterators. | 调试打印、文件I/O、`allclose`、nan/inf检查、广播辅助、通用迭代器。 |

---

## Directory Structure
## 目录结构

```
.
├── include/          # All header files                         # 所有头文件
│   ├── tensor.h
│   ├── compare_ops.h
│   ├── fft_ops.h
│   ├── indexing.h
│   ├── linalg_ops.h
│   ├── math_ops.h
│   ├── nn_ops.h
│   ├── random_ops.h
│   ├── reduce_ops.h
│   ├── search_ops.h
│   ├── shape_ops.h
│   └── utils.h
├── src/              # Implementation files                     # 实现文件
│   ├── tensor.c
│   ├── compare_ops.c
│   ├── fft_ops.c
│   ├── indexing.c
│   ├── linalg_ops.c
│   ├── math_ops.c
│   ├── nn_ops.c
│   ├── random_ops.c
│   ├── reduce_ops.c
│   ├── search_ops.c
│   ├── shape_ops.c
│   └── utils.c
├── tests/            # Unit tests (one per module)              # 单元测试（每个模块一个）
│   ├── test_*.c
│   └── runbuild_test_*.bat (Windows batch files to compile & run each test) # Windows批处理文件，用于编译和运行每个测试
└── README.md
```

---

## Testing
## 测试

Each module has a corresponding test file (e.g. `test_math_ops.c`).  
每个模块都有一个对应的测试文件（例如 `test_math_ops.c`）。

To build and run all tests, you can use the provided batch files (Windows) or write a simple Makefile.  
要构建并运行所有测试，可以使用提供的批处理文件（Windows）或编写一个简单的Makefile。

Example (using GCC on Linux):  
示例（在Linux上使用GCC）：
```bash
gcc -Iinclude -o test_math tests/test_math_ops.c src/*.c -lm
./test_math
```

All tests should pass with no errors.  
所有测试应无错误通过。

---

## License
## 许可证

This library is released under the **MIT License**.  
该库以**MIT许可证**发布。

---

## Contributing
## 贡献

Contributions are welcome! Please ensure that:
- Code follows the existing style.
- New functions are accompanied by appropriate tests.
- The library remains **platform‑independent** (no compiler‑specific extensions, no POSIX/Windows‑only calls).

欢迎贡献！请确保：
- 代码遵循现有风格。
- 新函数附有适当的测试。
- 该库保持**平台无关性**（不使用编译器特定扩展，不调用仅POSIX/Windows的API）。

---

## Acknowledgements
## 致谢

This library was inspired by NumPy, PyTorch, and various C tensor libraries.  
该库的灵感来自NumPy、PyTorch以及各种C语言张量库。

Special thanks to the open‑source community for providing reference implementations of numerical algorithms.  
特别感谢开源社区提供了数值算法的参考实现。

DeepSeek-R1
深度求索-R1
