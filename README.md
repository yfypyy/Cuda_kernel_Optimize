# Cuda_kernel_Optimize
这是一个用于记录cuda学习的仓库

---
<img width="452" height="328" alt="image" src="https://github.com/user-attachments/assets/6fa3c1b5-c54a-4ac5-8ad0-8dfd2da41f94" />

# GPU 矩阵乘法优化笔记

> 本文基于 `flex_gpu_learn` Wiki 上的一篇博客，记录了针对 **8800 G（Tesla）显卡** 使用多种优化方法后的 GFLOPs 变化图。  
> 在服务器上，我们使用性能更强的 **RTX 6000 显卡**，二者在硬件架构上存在差异。
<img width="438" height="159" alt="image" src="https://github.com/user-attachments/assets/88391b51-af0b-4383-9fb7-c32a503561c7" />

---

## 实验环境与限制

- 服务器上无 `sudo` 权限，GPU 权限不完整
- 无法使用 **Nsight Compute**，仅能使用 **Nsight System**(但是也可以scp报告回本地分析)
- 通过 **CUDA Event** 手动打印性能信息，这也是本次实验的核心手段之一

---

## 优化方法详解

### 1. Tiling（分块）

#### 原理

矩阵乘法 \( C[i, j] = \sum_k A[i, k] \times B[k, j] \) 中，\( C \) 的每个元素需要 \( A \) 的整行和 \( B \) 的整列。  
Tiling 通过将矩阵分块，减少对全局内存的重复访问：

- 将 \( C \) 划分为多个 tile（例如 \( \frac{1}{2} \) 行 × \( \frac{1}{2} \) 列）
- 每个 block 负责一个 tile 的计算
- 依次加载 \( A \) 和 \( B \) 的子块到共享内存，进行局部内积，累加结果

#### 图示思路

<img width="452" height="317" alt="image" src="https://github.com/user-attachments/assets/a4453b99-524e-4396-887e-316eeeadaabe" />


---

### 2. 额外优化手段

#### `__restrict__` 指针修饰

- 表示指针指向的内存区域不与任何其他 `__restrict__` 指针重叠
- 编译器可进行更激进的加载、缓存优化、指令重排和向量化

#### 循环边界处理（`-1` 的作用）

- 当矩阵维度 \( K \) 不是 tile 大小的整数倍时，不加 `-1` 会导致多计算一块
- 使用 `-1` 确保循环边界正确向下取整

#### 循环展开

- 减少循环控制指令（如计数、跳转）的开销
- 提升指令级并行性

#### 合并访问（Coalesced Access）

- GPU 的一次内存事务会连续加载一段地址
- 对于二维数组，**行遍历**比列遍历更高效
- 因此对矩阵 \( B \) 提前进行**转置**，以提高访问效率
<img width="447" height="697" alt="image" src="https://github.com/user-attachments/assets/86bdbb3a-79ae-42ef-ac8f-e65ebaf8140f" />

---

### 3. Bank Conflict 与 Shared Memory 优化
<img width="413" height="243" alt="image" src="https://github.com/user-attachments/assets/c57c7fec-3aeb-491e-9e20-c6bd125c3bde" />

- 假设共享内存定义为 `s[32][32]`，若 32 个线程访问同一列（索引 `tid * 32 % 32`），会发生 bank conflict
- 通过**添加 padding**（如 `s[32][33]`），使得索引变为 `tid * 33 % 33`，访问地址被打散，减少冲突
<img width="436" height="698" alt="image" src="https://github.com/user-attachments/assets/c6d8ec1e-1eaa-4833-a2cc-f88abfafca1c" />

---

### 4. 寄存器分块（Register Blocking）

#### 背景

内积计算：
```cpp
accu += A_shared[i] * B_shared[j];
```
- GPU 指令集（SASS）通常限制一条指令只能从 shared memory 读取一个操作数
- 编译器会拆分为：
  1. 从 shared memory 加载 B 到寄存器
  2. 执行 FMA（乘加）指令（A 来自 shared memory，B 来自寄存器）

#### 外积优化

- 将 \( B \) 的一行数据**提前加载到寄存器**中
- 循环中只从 shared memory 读取 \( A \)
- 代码变为：
  ```cpp
  accu += A_shared[i] * B_register;
  ```
- 减少 shared memory 访问次数，提升指令效率

> ⚠️ 注意：寄存器数量有限，过度使用会导致寄存器溢出，反而降低性能。
<img width="430" height="476" alt="image" src="https://github.com/user-attachments/assets/e0b82131-2d0e-44f9-9064-780540485b3d" />

---

### 5. 双缓冲（Double Buffering）

- 在当前计算使用一个 buffer 的同时，异步加载下一块数据到另一个 buffer
- 有效隐藏数据加载延迟，提升吞吐量

---

