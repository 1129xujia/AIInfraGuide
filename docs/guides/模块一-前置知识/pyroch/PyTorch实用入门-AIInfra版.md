---
title: "PyTorch 实用入门：先把模型跑对，再看性能"
description: "面向 AI Infra 初学者，用一个线性投影层学会 Tensor 检查、最小训练、模型推理、批处理计时和显存估算"
pubDate: 2026-11-12
category: "prerequisites"
order: 399
chapter: 4
tags: ["PyTorch", "AI Infra", "Tensor", "推理", "入门"]
---

# PyTorch 实用入门：先把模型跑对，再看性能

学习 AI Infra，先要能回答几个具体问题：

- 模型收到的数据形状对不对，放在哪里，占多少内存？
- 训练时，哪几行代码真正更新了参数？
- 推理时，怎样关闭不需要的梯度记录？
- 为什么逐条处理慢，合成一批可能更快？
- 出现 shape 报错、设备不一致或显存不足，先检查什么？

这份讲义用**一个线性投影层**回答这些问题。它是 Transformer 中 Q/K/V 投影和 FFN 的基础运算。我们缩小维度，让你能看懂和检查每一步。

这里的输入是人工生成的隐藏向量，实验用于学习运行机制与验证正确性，不具备语言理解能力。

**建议分两次学习：**先完成第 1～7 节，把训练和推理跑通；第二次完成第 8～10 节，练习排错与性能观察。

你本地已经配置好 `AI Infra (PyTorch CUDA)` 内核。双击项目根目录的 `start-pytorch-notebook.cmd`，或者在 VS Code 打开同名 Notebook 并选择该内核，再从上往下运行。

## 1. 先确认：代码会在哪个设备上运行？

CPU 是主机处理器；`cuda` 表示这里使用 NVIDIA GPU。先运行下面这个单元，后续都使用同一个 `device`。

只需修改 `use_gpu`，就能对照 CPU 和 GPU。固定随机种子，方便重复实验；它不保证所有设备的结果逐位一致。

```python
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

use_gpu = True
torch.manual_seed(42)
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("运行设备:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
```

**记住一个习惯：先看实际输出，不凭电脑有显卡就认定代码正在用 GPU。**

## 2. Tensor：先会看三个属性

把 Tensor 理解成“带有形状、数据类型和存放位置的数组”。

在大模型里，经常看到 `[B, N, D]`：

| 符号 | 含义 | 本例 |
|---|---|---|
| B | 一批有多少条请求 | 2 |
| N | 每条请求有多少个 token | 3 |
| D | 每个 token 用多少个数表示 | 4 |

```python
hidden_states = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)

print("shape:", hidden_states.shape)
print("dtype:", hidden_states.dtype)
print("device:", hidden_states.device)
print("第一条请求的第一个 token:", hidden_states[0, 0])

assert hidden_states.shape == (2, 3, 4)
```

这里 `hidden_states[0, 0]` 是一个 token 的四维表示，内容为 `[0, 1, 2, 3]`。

读陌生模型代码时，先在关键位置打印这三个属性，往往比逐行猜测更有效。

## 3. 一个模型到底做了什么？

先用最简单的模型：

```python
model = nn.Linear(4, 2).to(device)
output = model(hidden_states)

print("输入:", hidden_states.shape)
print("输出:", output.shape)
print("内部权重:", model.weight.shape)

assert output.shape == (2, 3, 2)
```

`nn.Linear(4, 2)` 的意思是：**把每个位置的 4 个特征变成 2 个特征**。

```text
输入：2 条请求 × 每条 3 个 token × 每个 token 4 维
                         ↓ Linear(4, 2)
输出：2 条请求 × 每条 3 个 token × 每个 token 2 维
```

它计算的是：

$$
y=xW^\top+b
$$

PyTorch 把权重存成 `[输出维度, 输入维度]`，所以这里 `model.weight` 是 `[2, 4]`。`model` 是一个可以调用的对象，它内部保存着可学习的权重和偏置。

**AI Infra 联系：**Q/K/V 投影也是类似的线性变换，只是实际维度更大。优化这种运算时，主要会遇到矩阵乘法、数据布局、精度和 GPU 利用率。

## 4. 给模型一个能检查对错的小任务

先生成 256 条四维输入，再用一个已知的规则得到二维目标：

```python
inputs = torch.randn(256, 4, device=device)
reference_weight = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, -1.0],
    [0.5, 0.5],
], device=device)
reference_bias = torch.tensor([0.1, -0.2], device=device)
targets = inputs @ reference_weight + reference_bias

print("训练输入:", inputs.shape)
print("参考输出:", targets.shape)
```

`@` 表示矩阵乘法，`*` 表示逐元素乘法，二者用途不同。

接下来，让模型学会接近这个已知规则。真实训练的目标可能来自数据标签；这里使用人工规则，是为了能直接检查模型是否学对。开发新算子时，也常用这种“参考结果对照”的方法验证正确性。

## 5. 训练：看懂这五步就能读懂基本流程

先不用记各种优化器和调度器。这里使用 SGD，它根据梯度调整参数。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model.train()

for step in range(101):
    optimizer.zero_grad()                         # 1. 清除上一轮梯度
    prediction = model(inputs)                    # 2. 计算预测
    loss = (prediction - targets).square().mean() # 3. 计算误差
    loss.backward()                               # 4. 计算参数的梯度
    optimizer.step()                              # 5. 更新参数

    if step % 25 == 0:
        print(f"step={step:3d}，loss={loss.item():.6f}")
```

读这段代码时，只抓住下面的关系：

| 名字 | 用普通话理解 |
|---|---|
| prediction | 模型目前给出的答案 |
| loss | 答案与目标差多少，这里用平均平方误差 |
| backward | 计算每个参数应该如何调整的依据 |
| optimizer.step | 按这个依据修改参数 |
| zero_grad | 清掉上一轮留下的梯度，因为梯度默认会累加 |

你应该看到 loss 明显下降。**参数在 `optimizer.step()` 中更新；`backward()` 负责算梯度。**

这里每 25 步才记录一次 loss。GPU 上的 `loss.item()` 会取回 CPU 标量，可能引入同步，不要在性能敏感的循环里无意地频繁调用。

## 6. 推理：输入一条请求，拿到模型结果

训练完成后，我们输入 `[1, 2, 3, 4]`。按前面的参考规则，正确输出应当接近：

$$
[1+3+0.5\times4+0.1,\quad 2-3+0.5\times4-0.2]=[6.1,\ 0.8]
$$

```python
model.eval()
request = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)

with torch.inference_mode():
    result = model(request)

print("模型输出:", result)
expected = torch.tensor([[6.1, 0.8]], device=device)
torch.testing.assert_close(result, expected, atol=0.01, rtol=0)
assert not result.requires_grad
```

两行常见代码各有作用：

- `model.eval()`：让模型中的相关层切换到评估行为。这个单独的 Linear 不会因此改变计算，但真实模型中的 Dropout 等会受到影响。
- `torch.inference_mode()`：推理时关闭不需要的梯度记录等开销。

**实际使用模型时，常把这两者一起使用；`eval()` 本身不等于关闭梯度。**

## 7. 保存参数，下次直接加载推理

不需要每次打开程序都重新训练。

```python
output_dir = Path("pytorch_simple_outputs")
output_dir.mkdir(exist_ok=True)
weight_path = output_dir / "linear.pt"
torch.save(model.state_dict(), weight_path)

loaded_model = nn.Linear(4, 2).to(device)
weights = torch.load(weight_path, map_location=device, weights_only=True)
loaded_model.load_state_dict(weights)
loaded_model.eval()

with torch.inference_mode():
    loaded_result = loaded_model(request)

torch.testing.assert_close(loaded_result, result)
print("参数已保存，重新加载后结果一致:", weight_path)
```

`state_dict()` 在这个模型中保存权重和偏置；模型的结构仍由 `nn.Linear(4, 2)` 定义。推理只需这些权重；完整恢复训练还涉及优化器等状态，放到之后再学。

到这里，你已经完成了：**准备数据 → 训练 → 推理 → 保存 → 加载 → 检查结果**。

## 8. 第一种实用能力：看到报错知道检查哪里

例如模型要求最后一维是 4，我们却传了 5：

```python
wrong_input = torch.randn(2, 5, device=device)

try:
    with torch.inference_mode():
        loaded_model(wrong_input)
except RuntimeError as error:
    print("预期中的 shape 报错:", str(error).splitlines()[0])

correct_input = torch.randn(2, 4, device=device)
with torch.inference_mode():
    correct_output = loaded_model(correct_input)
print("改正后的输出形状:", correct_output.shape)
assert correct_output.shape == (2, 2)
```

遇到错误，按这个表先做最小检查：

| 现象 | 先检查什么 | 常见处理 |
|---|---|---|
| 矩阵不能相乘 | 输入最后一维和层的输入维度 | 本例必须是 `[..., 4]` |
| CPU/CUDA 设备不一致 | 输入和模型参数的 device | 把模型与输入放在同一 device |
| dtype 不一致 | 输入和模型参数的 dtype | 例如统一为 float32 |
| CUDA out of memory | batch、序列长度，是否保存了大量 GPU Tensor | 先缩小 batch/序列长度，检查不再需要的引用 |

查看模型参数可以使用：

```python
parameter = next(loaded_model.parameters())
print("模型参数:", parameter.shape, parameter.dtype, parameter.device)
print("输入数据:", request.shape, request.dtype, request.device)
```

不要把 `empty_cache()` 当作通用 OOM 修复：它不会释放仍被对象引用的 Tensor。

## 9. 第二种实用能力：比较逐条推理和批量推理

假设服务收到 64 条请求。每条单独调用模型，会有 64 次 Python 调用和相应的算子调度；把它们合成一批，可以减少这些开销。

先验证两种方式的结果一致，再讨论快慢。

```python
requests = torch.randn(64, 4, device=device)

def run_one_by_one():
    pieces = [loaded_model(requests[i:i+1]) for i in range(len(requests))]
    return torch.cat(pieces, dim=0)

def run_batch():
    return loaded_model(requests)

with torch.inference_mode():
    serial_result = run_one_by_one()
    batch_result = run_batch()

torch.testing.assert_close(serial_result, batch_result, atol=1e-5, rtol=1e-5)
print("两种方式输出一致:", batch_result.shape)
```

`torch.cat(..., dim=0)` 将各条请求的结果沿第 0 维拼起来。

再测时间。下面函数只做三件事：**预热、重复运行、计算平均耗时**。GPU 运算通常异步执行，所以在计时边界同步，测量这批任务完成的端到端耗时。

```python
def measure_ms(function, repeat=20):
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(repeat):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (perf_counter() - start) * 1000 / repeat

with torch.inference_mode():
    serial_ms = measure_ms(run_one_by_one)
    batch_ms = measure_ms(run_batch)

print(f"逐条处理 64 条请求: {serial_ms:.3f} ms")
print(f"批量处理 64 条请求: {batch_ms:.3f} ms")
print(f"批量方式的实验吞吐量: {64 / (batch_ms / 1000):.0f} 条/秒")
```

这里的吞吐量是“一次处理的请求数 / 这次处理的时间”。它只代表当前设备和这个小模型的实验结果，不包含生产服务里的排队、网络和文本预处理。

**AI Infra 的基本工作方式：先验证结果相同，再比较性能。** 小模型不一定在 GPU 上比 CPU 快；改变 batch size，再测一次，比记住某个固定加速比更有用。

## 10. 第三种实用能力：会算数据占用多少内存

Tensor 的基本数据量：

$$
\text{字节数}=\text{元素个数}\times\text{每个元素的字节数}
$$

| dtype | 每个元素字节数 |
|---|---:|
| float32 | 4 |
| float16 | 2 |
| bfloat16 | 2 |

```python
sample = torch.empty(2, 128, 64, dtype=torch.float32, device=device)
tensor_bytes = sample.numel() * sample.element_size()
print(f"这块 Tensor 的数据量: {tensor_bytes / 1024:.0f} KiB")
assert tensor_bytes == 65536

if device.type == "cuda":
    print(f"PyTorch 当前活跃分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
```

这块 Tensor 是 `2 × 128 × 64 × 4 = 65536` 字节，也就是 64 KiB。它的数据量不等于进程的全部显存；模型、其他张量、运行时等也会占用空间。

### 选读：用同一个思路估算 KV Cache

对常见的、未压缩且 K/V 维度相同的 KV Cache：

$$
\text{字节数}=2\times\text{层数}\times B\times N\times
\text{KV头数}\times\text{每头维度}\times\text{每元素字节数}
$$

第一个 2 表示 Key 和 Value 各一份。GQA 要代入 **KV 头数**，不是 Q 头数。

```python
layers, batch, tokens = 32, 1, 2048
kv_heads, head_dim, bytes_per_element = 8, 64, 2
kv_cache_bytes = (
    2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_element
)
print(f"这个配置的 KV Cache 理论数据量: {kv_cache_bytes / 1024**2:.0f} MiB")
assert kv_cache_bytes == 128 * 1024**2
```

结果是 128 MiB，只包括上述 KV 数据。batch 或缓存序列长度翻倍时，这部分数据量也翻倍。

## 学完后，动手改三处

1. 将 `hidden_states` 改为 `[4, 16, 4]`，先预测 Linear 输出形状，再验证。
2. 将请求数从 64 改为 256，同时修改计时输出里的请求数和吞吐量分子，比较结果。
3. 将 KV Cache 例子的 batch 从 1 改为 4，先手算数据量；同步调整相应检查值，再运行验证。

检查自己是否能独立完成：

- [ ] 看到输入知道 shape、dtype、device 分别是什么。
- [ ] 能解释 Linear 的输入维度和输出维度。
- [ ] 能指出 backward 与 optimizer.step 的分工。
- [ ] 能推理、保存权重，再加载得到相同结果。
- [ ] 遇到 shape/device 报错知道先打印哪里。
- [ ] 能正确计时，并解释为什么要预热和同步。
- [ ] 能估算 Tensor 与普通 KV Cache 的数据量。

先把这些任务做熟，再学习 Dataset/DataLoader、混合精度、profiler 和分布式训练，会更容易把新概念放到实际流程里。

进阶参考：[原完整讲义](PyTorch框架入门.md)、[PyTorch 官方入门教程](https://pytorch.org/tutorials/beginner/basics/intro.html)。
