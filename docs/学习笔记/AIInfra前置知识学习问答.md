# AI Infra 前置知识学习问答

整理日期：2026-11-12

本文根据学习过程中的 26 个问题整理，覆盖 Python 对象模型、并发与通信、C++ 资源管理、Linux、Transformer 和 GPU。重复追问合并到相应主题，保留关键代码、数值例子和容易混淆的边界。

Python 示例以 Python 3.10+ 为基础；涉及 GIL 的说明针对启用 GIL 的常见 CPython 构建。C++ 示例主要使用 C++17。Linux 命令以常见 GNU/Linux 环境为例。

## 目录

- [Python 对象与函数](#python-对象与函数)
- [并发与通信](#并发与通信)
- [C++ 与编译构建](#c-与编译构建)
- [Linux 进程与资源查看](#linux-进程与资源查看)
- [Transformer 与推理服务](#transformer-与推理服务)
- [GPU 架构与计算量](#gpu-架构与计算量)
- [易混淆概念速查](#易混淆概念速查)
- [自我检验清单](#自我检验清单)
- [参考原文](#参考原文)

## Python 对象与函数

### 1. 为什么 append 能改变调用方的列表，重新赋值却不能？

Python 中，名字绑定到对象。调用函数时，形参名字和调用方的名字最初引用同一个对象，但它们仍然是两个独立的名字。

```python
def mutate(xs: list[int]) -> None:
    xs.append(1)

def rebind(xs: list[int]) -> None:
    xs = [1, 2]

data: list[int] = []
mutate(data)
print(data)  # [1]

rebind(data)
print(data)  # [1]
```

进入函数时：

```text
data ──┐
       ├──→ 同一个列表
xs   ──┘
```

- `xs.append(1)` 修改列表对象，因此通过 `data` 也能看到变化。
- `xs = [1, 2]` 创建新列表，再改变局部名字 `xs` 的绑定；`data` 仍然引用原来的列表。

**修改对象会影响其他引用它的名字；重新赋值只改变被赋值的那个名字。**

### 2. xs[:] 是什么写法？为什么切片赋值能修改原列表？

切片的一般格式为：

```python
xs[start:stop:step]
```

开始位置包含在内，结束位置不包含在内。省略开始、结束位置表示覆盖整个序列，默认步长为 1。

```python
xs = [10, 20, 30, 40]

print(xs[1:3])  # [20, 30]
print(xs[:2])   # [10, 20]
print(xs[2:])   # [30, 40]
print(xs[:])    # [10, 20, 30, 40]
```

读取列表切片和对列表切片赋值，是两种操作：

```python
ys = xs[:]       # 创建新列表：浅拷贝
xs[:] = [1, 2]   # 修改原列表：替换它的全部元素
xs = [3, 4]      # 重新绑定名字：指向另一个列表
```

浅拷贝只复制外层列表，内部可变对象仍可能共享。切片赋值允许替换前后的元素数量不同。

### 3. append 这类操作都是直接修改原对象吗？

对列表而言，`append()` 是原地操作：

```python
a = [1, 2]
b = a
a.append(3)

print(b)       # [1, 2, 3]
print(a is b)  # True
```

但是否修改原对象，要看具体类型和方法的约定：

| 列表操作 | 是否修改原列表 | 返回值 |
|---|---|---|
| `append(x)` | 是，追加一个元素 | `None` |
| `extend(xs)` | 是，追加多个元素 | `None` |
| `insert(i, x)` | 是，插入元素 | `None` |
| `pop()` | 是，删除元素 | 被删除的元素 |
| `sort()`、`reverse()` | 是 | `None` |
| `clear()` | 是，清空元素 | `None` |
| `copy()` | 否 | 新的浅拷贝列表 |
| `sorted(a)` | 否 | 排序后的新列表 |

不要写：

```python
a = a.append(3)
```

它会先修改列表，再把 `append()` 返回的 `None` 赋给 `a`。

### 4. 迭代器、生成器和 yield 如何工作？

- **可迭代对象**：可以通过 `iter(obj)` 获取迭代器。
- **迭代器**：通过 `next(it)` 逐个产生元素，并保存遍历状态。
- **生成器函数**：包含 `yield` 的函数；调用后返回生成器对象，它也是迭代器。

```python
from collections.abc import Iterator

def read_batches(path: str, batch_size: int) -> Iterator[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    batch: list[str] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            batch.append(line.rstrip("\n"))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

假设文件中有 `a、b、c、d、e` 五行：

```python
batches = read_batches("data.txt", 2)  # 此时尚未执行函数体

print(next(batches))  # ['a', 'b']
print(next(batches))  # ['c', 'd']
print(next(batches))  # ['e']
# 再次 next(batches) 会抛出 StopIteration
```

执行到 `yield batch` 时，当前列表被交给调用方，函数暂停。下一次请求元素时，从 `yield` 后面继续执行。

这里的 `batch = []` 很重要：它让局部名字指向新列表，不会修改已经交出去的那一批。换成 `batch.clear()` 会清空调用方可能仍在保存的同一个列表。

通常使用 `for` 自动获取元素并处理结束：

```python
for batch in read_batches("data.txt", 2):
    print(batch)
```

需要记住：

1. 同一个生成器消费到末尾后不会自动重置；要重新调用函数创建新的生成器。
2. 函数体的异常也延迟到迭代时发生，例如找不到文件、批次大小不合法。
3. `list(read_batches(...))` 会收集全部批次。节省内存的前提是逐批处理，而不是一直保存所有批次。

### 5. 装饰器 timed 是怎样包住原函数的？

装饰器接收函数，返回一个包装后的可调用对象。计时装饰器可以在不修改原函数体的情况下，添加调用前后的行为。

```python
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def timed(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (perf_counter() - start) * 1_000
            print(f"{fn.__name__}: {elapsed_ms:.3f} ms")
    return wrapper

@timed
def preprocess(values: list[float]) -> list[float]:
    return [value * 2 for value in values]
```

`@timed` 等价于：

```python
preprocess = timed(preprocess)
```

执行装饰时，`timed` 创建并返回 `wrapper`，还不会运行原函数。之后调用 `preprocess(...)`，实际进入 `wrapper`；它通过闭包保存的 `fn` 调用原函数。

```text
调用 preprocess → wrapper 开始计时 → 原函数执行 → finally 打印耗时 → 返回结果
```

| 写法 | 作用 |
|---|---|
| `return wrapper` | 返回包装函数本身 |
| `*args`、`**kwargs` | 接收位置参数和关键字参数，并原样转发 |
| `finally` | 返回前或异常传播前执行计时输出 |
| `@wraps(fn)` | 保留原函数名字、文档字符串等元信息，并建立 `__wrapped__` 关联 |
| `ParamSpec P` | 描述原函数的参数形式 |
| `TypeVar R` | 描述原函数的返回类型 |
| `Callable[P, R]` | 参数形式为 P、返回类型为 R 的可调用对象 |

原函数抛出异常时也会打印耗时，异常仍会继续向外传播。类型标注主要服务于静态检查和 IDE，不会自动执行运行时类型验证。

## 并发与通信

### 6. Lock、global 和 with lock 分别有什么作用？

```python
from threading import Lock

lock = Lock()
counter = 0

def increment() -> None:
    global counter
    with lock:
        counter += 1
```

- `global counter`：让赋值更新模块中的名字 `counter`。
- `lock`：多个线程共同使用的一把互斥锁。
- `with lock`：进入时获取锁，离开时自动释放锁，包括异常退出。

更新计数器可以分解为：

```text
读取旧值 → 计算新值 → 写回
```

如果这些步骤交错，可能覆盖其他任务的更新。不要依赖 GIL 或某个解释器版本的执行细节来保证复合操作的正确性。

锁保护的是遵守同一把锁的操作；它不会自动给变量的所有访问加锁。上述代码本身只定义了函数，还没有创建线程或执行加法。

### 7. 多进程为什么能绕开 GIL？数据搬运成本在哪里？

启用 GIL 的常见 CPython 中，一个进程内通常只有一个线程能执行 Python 字节码。多个进程有各自的解释器和 GIL，因此可以在多个 CPU 核心上并行执行纯 Python 计算。

```text
进程 A → 解释器 A、GIL A → CPU 核心 1
进程 B → 解释器 B、GIL B → CPU 核心 2
```

但不同进程通常不直接共享普通 Python 对象。使用普通进程池传递参数、结果时，往往需要：

```text
主进程对象 → 序列化 → 进程间传输 → 反序列化 → 子进程对象
```

成本包括创建进程、调度任务、序列化、数据传输、额外内存和同步。`fork` 可能使用写时复制，并不代表所有内存立即复制；进程池任务的普通参数仍通常通过序列化传送。

| 任务形态 | 多进程的适合程度 |
|---|---|
| 传一个整数，计算大量质数，再返回一个计数 | 数据少、计算多，比较合适 |
| 传一个巨大列表，只取第一个元素 | 搬运可能远比计算昂贵 |
| 大型数组反复跨进程传递 | 需要考虑共享内存、减少复制 |

可采用的优化：增大每个任务的工作量、复用进程池、减少输入输出数据、传文件路径或共享内存描述信息。是否获得加速，最终需要测量。

### 8. 协程和 asyncio 是怎样实现并发的？

协程是一种能暂停、保留状态、稍后继续执行的任务。在常见单线程 `asyncio` 程序中，事件循环调度多个协程，让它们在等待期间交替推进。

```python
import asyncio

async def worker(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return name

async def main() -> None:
    results = await asyncio.gather(
        worker("a", 0.2),
        worker("b", 0.1),
    )
    print(results)

asyncio.run(main())
```

| 时间 | 发生什么 |
|---|---|
| 约 0 秒 | A 等待 0.2 秒；B 等待 0.1 秒 |
| 约 0.1 秒 | B 恢复并完成 |
| 约 0.2 秒 | A 恢复并完成；main 打印结果 |

总耗时约为 0.2 秒。输出为 `['a', 'b']`，因为 `gather` 按传入顺序返回结果，而不是按完成顺序。

- 调用 `async def` 定义的函数，先得到协程对象，不会立即执行函数体。
- `gather` 会把传入的协程安排为并发任务；`asyncio.run` 负责启动事件循环并执行入口协程。
- `await` 是可能暂停的位置；如果操作已经能够立即完成，不一定切换任务。
- `await asyncio.sleep(...)` 会挂起当前协程；`time.sleep(...)` 会阻塞事件循环所在的线程。
- `async def` 不会自动把普通阻塞调用和 CPU 计算变成异步操作。

已有的阻塞 I/O 函数可以放在线程中：

```python
result = await asyncio.to_thread(blocking_function, argument)
```

纯 Python 的大量 CPU 计算通常更适合进程池或原生计算库。

### 9. 只有线程会出现并发问题吗？

线程、进程、协程都可能遇到并发问题。关键是多个任务是否操作同一个可修改资源，以及操作是否可能交错。

| 执行方式 | 常见共享资源 | 典型协调方式 |
|---|---|---|
| 多线程 | 同一进程中的变量、容器 | `threading.Lock` |
| 多进程 | 共享内存、文件、数据库 | 进程锁、文件锁、事务 |
| 协程 | 同一事件循环中的变量、容器 | `asyncio.Lock` |

即使只有一个线程，协程也可能发生逻辑竞态：

```python
import asyncio

counter = 0

async def increment():
    global counter
    old = counter
    await asyncio.sleep(0)
    counter = old + 1

async def main():
    await asyncio.gather(increment(), increment())
    print(counter)  # 1，两个任务都先读取到了 0

asyncio.run(main())
```

两个任务没有同时执行，但“读旧值—暂停—写新值”发生交错，仍然丢失了一次更新。

可将上例的函数替换为：

```python
lock = asyncio.Lock()

async def increment():
    global counter
    async with lock:
        old = counter
        await asyncio.sleep(0)
        counter = old + 1
```

此时两个完整更新依次进行，结果为 2。在普通单线程事件循环中，连续执行、没有暂停点的 `counter += 1` 不会被另一个协程从中间插入执行；跨 `await` 使用旧状态时尤其需要留意竞态。

### 10. 线程和进程有哪些相同、不同的通信方式？

两者都可以共享数据、传消息、发通知；主要差别在于线程共享地址空间，进程通常拥有独立地址空间。

| 方式 | 线程之间 | 进程之间 |
|---|---|---|
| 普通变量、列表、字典 | 可以访问同一个对象 | 通常各自独立，修改不自动同步 |
| 队列 | 常传递同一对象的引用 | 普通对象通常序列化后传输 |
| 共享内存 | 已共享进程内存 | 需要显式建立共享区域 |
| 管道 Pipe | 可以使用，但较少这样做 | 常见通信方式 |
| Socket | 可以使用 | 可以使用；TCP 等还支持跨机器 |
| 文件、数据库 | 可以使用，需要协调并发访问 | 同样需要文件锁、事务等 |

Python 中常用的对应关系：

| 用途 | 线程 | 进程 |
|---|---|---|
| 队列 | `queue.Queue` | `multiprocessing.Queue` |
| 锁 | `threading.Lock` | `multiprocessing.Lock` |
| 事件通知 | `threading.Event` | `multiprocessing.Event` |
| 条件变量 | `threading.Condition` | `multiprocessing.Condition` |
| 信号量 | `threading.Semaphore` | `multiprocessing.Semaphore` |

线程队列的接收方可能拿到同一个可变对象；进程队列传递普通列表时，接收方通常拿到反序列化重建的独立列表。

`multiprocessing.shared_memory`、`Value`、`Array` 可以显式共享数据；`Manager` 提供代理对象，操作通常需要与管理进程通信。

**通信解决数据如何传递，同步解决操作如何协调。** 各进程分别创建独立的 `threading.Lock`，不能协调它们对同一个文件的写入。

## C++ 与编译构建

### 11. RAII 为什么能把资源绑定到对象生命周期？

RAII（Resource Acquisition Is Initialization）的做法是：构造时获取资源，析构时释放资源。资源可以是内存、文件句柄、锁、网络连接、GPU 显存等。

```cpp
#include <cstdio>
#include <stdexcept>

class File {
public:
    explicit File(const char* path)
        : handle_(std::fopen(path, "rb")) {
        if (handle_ == nullptr) {
            throw std::runtime_error("failed to open file");
        }
    }

    ~File() {
        std::fclose(handle_);
    }

    File(const File&) = delete;
    File& operator=(const File&) = delete;

private:
    std::FILE* handle_;
};

void load(bool valid) {
    File file("data.bin");
    if (!valid) {
        throw std::runtime_error("invalid data");
    }
}
```

`file` 是成功构造的局部对象。正常离开作用域、提前 `return`，或者异常传播导致栈展开时，都会执行它的析构函数，关闭文件。

禁止复制是为了避免两个对象直接复制同一个句柄，然后重复关闭。这里的教学类没有实现移动操作。

常用标准库 RAII 类型：

| 类型 | 管理的资源 |
|---|---|
| `std::vector`、`std::string` | 动态内存 |
| `std::fstream` | 文件 |
| `std::unique_ptr` | 独占管理的对象 |
| `std::lock_guard` | 互斥锁的持有与释放 |

```cpp
{
    std::lock_guard<std::mutex> guard(mutex);
    ++counter;
}  // guard 析构，自动解锁
```

Python 的 `with` 和 C++ RAII 都用于明确资源释放边界，但机制不同；不能依赖 Python 对象的析构时机来替代 `with`。

### 12. shared_ptr 的循环引用为什么会泄漏？

`shared_ptr` 通过强引用计数管理对象，计数降到 0 时销毁对象。

```cpp
#include <iostream>
#include <memory>

struct B;

struct A {
    std::shared_ptr<B> b;
    ~A() { std::cout << "A 被销毁\n"; }
};

struct B {
    std::shared_ptr<A> a;
    ~B() { std::cout << "B 被销毁\n"; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();
        a->b = b;
        b->a = a;

        std::cout << a.use_count() << " "
                  << b.use_count() << "\n";
    }
    std::cout << "已离开作用域\n";
}
```

输出只有：

```text
2 2
已离开作用域
```

局部变量销毁后，A 和 B 仍相互持有：

```text
A ──shared_ptr──→ B
↑                │
└──shared_ptr────┘
```

双方强引用计数都停在 1，因此没有触发析构。

把其中一个方向改成 `weak_ptr`：

```cpp
struct B {
    std::weak_ptr<A> a;
    ~B() { std::cout << "B 被销毁\n"; }
};
```

`weak_ptr` 不增加强引用计数，不延长目标对象的生命周期。访问前通过 `lock()` 获取临时 `shared_ptr`：

```cpp
if (auto owner = b->a.lock()) {
    // A 仍存在，可以通过 owner 访问
}
```

如果 A 已经销毁，`lock()` 返回空的 `shared_ptr`。这里的 `lock()` 用于尝试取得对象所有权，不是互斥锁操作。

### 13. 拷贝和移动有什么区别？

以 `std::vector<int>` 为例，拷贝复制元素到独立存储，移动构造通常转移缓冲区的所有权。

```cpp
#include <utility>
#include <vector>

std::vector<int> a = {1, 2, 3};
std::vector<int> b = a;             // 拷贝构造
std::vector<int> c = std::move(a);  // 移动构造
```

```text
拷贝后：a → 缓冲区①，b → 独立的缓冲区②
移动后：c → 原缓冲区①，a 仍是有效的 vector 对象
```

对标准库的这类对象，移动后通常应按“有效但状态未指定”使用，不依赖它保留原内容，也不统一假设为空。可以重新赋值：

```cpp
a = {4, 5};
```

**`std::move` 本身不搬运资源。** 它把表达式转换为允许移动的形式，实际操作由类型的移动构造或移动赋值完成。没有适用的移动操作时，也可能发生拷贝。

```cpp
std::vector<int> d = b;             // 拷贝构造：创建 d
std::vector<int> e = std::move(b);  // 移动构造：创建 e

d = c;             // 拷贝赋值：更新已有对象 d
d = std::move(c);  // 移动赋值：更新已有对象 d
```

`unique_ptr` 禁止拷贝，但允许移动：

```cpp
auto p1 = std::make_unique<int>(42);
// auto p2 = p1;           // 编译错误
auto p2 = std::move(p1);   // p2 接管对象，p1 变为空
```

`unique_ptr` 移动后变空是明确保证。一般自定义类型的移动后状态，则需要看它自己的接口约定。

类若只包含标准 RAII 成员，通常无需手写析构、复制、移动操作，这就是 **Rule of Zero**。

### 14. const float* 和 float* const 有什么区别？

```cpp
float a = 1.0f;
float b = 2.0f;

const float* p1 = &a;
// *p1 = 3.0f;  // 错误：不能通过 p1 修改值
p1 = &b;        // 正确：可以改变指向

float* const p2 = &a;
*p2 = 3.0f;     // 正确：可以修改指向的值
// p2 = &b;     // 错误：不能改变指向
```

| 声明 | 能改指向吗 | 能通过指针改值吗 |
|---|---|---|
| `const float* p1` | 能 | 不能 |
| `float* const p2` | 不能 | 能 |
| `const float* const p3` | 不能 | 不能 |

`const float*` 限制的是通过这个指针修改数据；如果原变量本身不是常量，仍可以通过原变量修改它。指针必须有有效指向后才能解引用。

### 15. 成员函数末尾的 const 有什么作用？

```cpp
#include <cstddef>

class TensorView {
public:
    std::size_t size() const {
        // size_ = 100;  // 错误：不能直接修改普通成员
        return size_;
    }

    void resize(std::size_t value) {
        size_ = value;
    }

private:
    std::size_t size_ = 10;
};
```

末尾的 `const` 限制函数通过当前对象进行修改，使普通成员可以读取、不能直接赋值。它不表示返回值是常量。

```cpp
const TensorView view;
auto n = view.size();  // 正确：const 对象可以调用 const 成员函数
n = 20;               // 正确：返回的是独立的值
// view.resize(20);   // 错误：不能调用这个非 const 成员函数
```

普通非 const 对象也能调用 `size()`。`mutable` 成员等有特殊规则；成员函数的 `const` 也不自动提供线程安全。

### 16. 词法分析、语法分析、语义分析分别做什么？

以 `int x = 3 + 5;` 为例：

| 阶段 | 做什么 | 典型产物 |
|---|---|---|
| 词法分析 | 把字符识别为关键字、标识符、字面量、符号等 | Token 序列 |
| 语法分析 | 根据语法规则，把 Token 组织成声明、表达式等结构 | 语法树、AST |
| 语义分析 | 检查名字绑定、作用域、类型、访问限制等规则 | 带类型、名字关联等信息的结构 |

词法分析得到：

```text
int | x | = | 3 | + | 5 | ;
```

语法分析得到类似结构：

```text
变量声明
├── 类型：int
├── 名字：x
└── 初始值：加法表达式
    ├── 3
    └── 5
```

不同错误的例子：

```cpp
int x = "hello;    // 词法问题：字符串未闭合
int x = 3 + ;      // 语法问题：缺少右操作数
int x = y + 1;     // 若 y 未声明，属于语义问题
int x = "hello";   // 语义问题：初始化类型不匹配
```

语法分析还会处理运算符优先级，例如把 `3 + 5 * 2` 组织成 `3 + (5 * 2)`。语义分析验证编程语言规则，不保证程序符合业务意图。

### 17. CMakeLists.txt 的作用是什么？

它描述项目如何构建：编译哪些源文件、生成什么目标、使用哪个语言标准、去哪里找头文件、链接哪些库。

```cmake
cmake_minimum_required(VERSION 3.20)
project(hello LANGUAGES CXX)

add_executable(app main.cpp helper.cpp)
target_compile_features(app PRIVATE cxx_std_17)
```

`app` 是构建目标（target）。其他常见指令包括 `add_library`、`target_include_directories`、`target_link_libraries`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

- 第一步读取当前目录的配置，在 `build` 中生成构建文件；`CMAKE_BUILD_TYPE` 主要用于单配置生成器。
- 第二步调用底层构建工具；`--config Release` 用于选择多配置生成器的 Release 配置。

```text
CMakeLists.txt
    ↓ CMake 配置、生成
Make / Ninja / MSBuild 等构建文件
    ↓ 构建工具调用编译器、链接器
可执行程序或库
```

CMake 负责组织构建；具体编译由 GCC、Clang、MSVC 等编译器完成。

## Linux 进程与资源查看

### 18. top、free、ps 分别怎么用？

| 命令 | 用途 |
|---|---|
| `top` | 持续观察 CPU、内存和进程变化 |
| `free` | 查看整台机器的内存和 Swap |
| `ps` | 查看进程列表、PID、父进程、启动命令等快照 |

```bash
top
top -p 1234       # 只观察 PID 1234
top -H -p 1234    # 展开观察这个进程中的线程
```

进入 `top` 后：

| 按键 | 作用 |
|---|---|
| `P` | 按 CPU 使用率排序 |
| `M` | 按内存占用排序 |
| `1` | 显示各逻辑 CPU 的使用情况 |
| `q` | 退出 |

重点字段：

| 字段 | 含义 |
|---|---|
| `PID` | 进程编号 |
| `%CPU` | CPU 使用率 |
| `%MEM` | 占主机物理内存的比例 |
| `RES` | 驻留物理内存大小 |
| `VIRT` | 虚拟地址空间大小，不能直接当作实际物理内存占用 |
| `S` | 状态，如 R 运行、S 休眠、Z 僵尸 |
| `COMMAND` | 程序名称或启动命令 |

默认常见显示模式下，进程的 `%CPU` 可以超过 100%；400% 通常表示约四个逻辑核心的计算占用。

```bash
free -h
free -h -s 2     # 每 2 秒输出一次，Ctrl+C 结束
```

- `total`：总内存。
- `used`：已用内存，具体计算口径可能随工具版本不同。
- `free`：完全空闲的内存。
- `buff/cache`：缓冲和缓存，其中一部分可回收。
- `available`：估计无需使用 Swap 即可提供给新应用的内存。

判断还有多少内存可用，通常优先看 `available`。Linux 会利用空闲内存缓存文件，所以 `free` 小不一定表示内存紧张。

```bash
ps aux
ps -ef
ps -p 1234 -f
ps -p 1234 -o pid,ppid,stat,pcpu,pmem,rss,etime,args

# 内存占用最大的 10 个进程，加上 1 行表头
ps -eo pid,comm,pcpu,pmem,rss --sort=-rss | head -n 11
```

`aux` 前面没有 `-`。`ppid` 是父进程编号，`rss` 通常以 KiB 为单位，`etime` 是启动后经过的时间，`args` 是完整命令和参数。

这些命令主要查看主机资源；查看 GPU 状态、显存和 GPU 进程，通常使用 `nvidia-smi`。

### 19. ps 和 top 的核心区别是什么？

| 对比 | ps | top |
|---|---|---|
| 展示方式 | 输出一次快照后退出 | 持续刷新，默认交互运行 |
| 适合的问题 | 进程是谁、父进程是谁、怎么启动的 | 谁持续占 CPU、内存是否不断增长 |
| 常见使用方式 | 参数筛选、管道、脚本 | 交互排序、连续观察 |
| CPU 指标 | 通常是进程存活期间的平均使用率 | 通常是最近采样区间的使用率 |

因此，同一个进程在两者中的 `%CPU` 可能不同。常用流程是先用 `ps` 找到目标 PID，再用 `top -p PID` 持续观察。

## Transformer 与推理服务

### 20. vLLM 是什么？

vLLM 是开源的大语言模型推理与服务引擎，用于高效运行训练好的模型，并为多个请求提供文本生成服务。

```text
用户问题 → vLLM 接收并调度请求 → GPU 执行模型 → 返回生成的 token
```

它的主要能力包括：

- **KV Cache 管理**：保存历史 token 的注意力 Key、Value，减少重复计算；PagedAttention 采用分块管理，减少缓存管理中的显存浪费。
- **连续批处理**：动态组织请求参与计算，在请求完成后继续加入新请求，提高资源利用率。
- **多卡并行**：让大模型跨多张 GPU 运行。
- **服务接口**：提供兼容 OpenAI 接口格式的服务，便于应用接入。

| 组件 | 分工 |
|---|---|
| Qwen、Llama 等模型 | 提供模型结构和训练好的权重 |
| PyTorch | 提供通用张量计算和深度学习能力 |
| vLLM | 调度推理、管理缓存、提供服务 |
| CUDA、GPU | 提供底层加速计算与硬件执行能力 |

vLLM 属于 AI Infra 的模型推理与服务层。常见优化指标是延迟、吞吐量、显存占用和并发容量。

### 21. Encoder–Decoder 与 Decoder-only 有什么区别？

Encoder–Decoder 使用两个模块分别处理输入序列和输出序列：

```text
输入文本 → Encoder → 输入的上下文表示
                          ↓ Cross-Attention
已生成文本 ───────────→ Decoder → 下一个 token
```

例如翻译“我爱你”：Encoder 编码完整输入；Decoder 在已经生成 `I love` 后，结合输入表示预测 `you`。

| 位置 | 注意力方式 | 信息来源 |
|---|---|---|
| Encoder 内部 | 双向 Self-Attention | 完整输入序列 |
| Decoder 内部 | 因果 Self-Attention | 当前及之前的输出位置 |
| Decoder 访问 Encoder | Cross-Attention | Q 来自 Decoder，K、V 来自 Encoder 输出 |

典型模型有原始 Transformer、T5、BART。

Decoder-only 将提示词和生成内容组织在同一条序列中：

```text
[指令 + 输入文本 + 已生成文本] → 因果 Self-Attention → 下一个 token
```

例如：

```text
请翻译成英文：
我爱你
英文：I love
```

模型根据前文继续生成 `you`。典型模型有 GPT、Llama、Qwen。

| 对比 | Encoder–Decoder | Decoder-only |
|---|---|---|
| 输入和输出 | 两条序列、两个模块 | 通常拼在一条序列中 |
| 输入侧的注意力 | Encoder 通常双向读取输入 | 典型 GPT 式结构采用因果注意力 |
| 获取输入信息 | 通过 Cross-Attention | 通过同一序列内的 Self-Attention |
| 常见应用 | 翻译、摘要等输入到输出的转换 | 通用生成、对话、代码生成 |

Decoder-only 同样会形成输入的上下文表示。两种结构生成时通常都逐 token 自回归；训练时则可以用因果 Mask 并行计算多个输出位置。

**Encoder/Decoder 描述模型结构，Prefill/Decode 描述推理阶段。** Decoder-only 也有 Prefill 和 Decode 阶段。

### 22. d_k 是否等于 d_model？缩放到底除以什么？

- `d_model`：模型隐藏表示的维度。
- `d_k`：一次注意力点积中，Q、K 向量的维度。

在文档的单头示例中，Q、K 投影后仍为 `d_model` 维，因此 `d_k = d_model`。这是示例的设定，一般单头投影也可以选择其他维度。

标准多头注意力通常令：

$$
d_k = \frac{d_{\text{model}}}{h}
$$

当 `d_model = 512`、`h = 8` 时，每个头的 Q、K 是 64 维，所以 `d_k = 64`。

正确的缩放公式是：

$$
S_{\text{scaled}} = \frac{QK^\top}{\sqrt{d_k}}
$$

因此，每个头应除以 $\sqrt{64}=8$，而不是除以 64 或 $\sqrt{512}$。

在 Q、K 分量独立、零均值、单位方差等简化假设下，点积方差随 `d_k` 增长；除以 $\sqrt{d_k}$ 可以稳定其量级，减少 Softmax 过度饱和。这是建模动机，不是实际训练中方差永远精确等于 1 的保证。

### 23. 用具体数字完整计算一次 MHA

MHA 让同一批 token 在多个可学习的投影空间中分别计算注意力，然后拼接并混合各头输出。

设定：

$$
N=3,\qquad d_{\text{model}}=4,\qquad h=2,\qquad d_k=d_v=2
$$

输入矩阵每行对应一个 token：

$$
X=
\begin{bmatrix}
1&0&1&0\\
1&0&0&1\\
0&1&1&0
\end{bmatrix}
\quad\leftarrow
\begin{matrix}
t_1\\t_2\\t_3
\end{matrix}
$$

先计算投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

为了手算，刻意设 $W_Q=W_K=W_V=I_4$，所以 `Q = K = V = X`。这只是本算例的特殊设置，真实模型中的投影通常不同。

把投影后的 4 维拆成两个 2 维头：

$$
Q_1=K_1=V_1=
\begin{bmatrix}
1&0\\
1&0\\
0&1
\end{bmatrix}
$$

$$
Q_2=K_2=V_2=
\begin{bmatrix}
1&0\\
0&1\\
1&0
\end{bmatrix}
$$

**两个头都处理全部 3 个 token。** 变化的是特征维度，不是序列被切成两段。

分别计算原始分数：

$$
S_1=Q_1K_1^\top=
\begin{bmatrix}
1&1&0\\
1&1&0\\
0&0&1
\end{bmatrix}
$$

$$
S_2=Q_2K_2^\top=
\begin{bmatrix}
1&0&1\\
0&1&0\\
1&0&1
\end{bmatrix}
$$

每个头都产生一个 $3\times3$ 矩阵。第 i 行表示 token i 对全部 token 的分数。

跟踪 $t_2$，即第二行。缩放并按行 Softmax：

$$
\operatorname{softmax}\left(
\left[\frac1{\sqrt2},\frac1{\sqrt2},0\right]
\right)
\approx[0.4011,\ 0.4011,\ 0.1978]
$$

$$
\operatorname{softmax}\left(
\left[0,\frac1{\sqrt2},0\right]
\right)
\approx[0.2483,\ 0.5035,\ 0.2483]
$$

| 对 t₂ 而言 | 关注 t₁ | 关注 t₂ | 关注 t₃ |
|---|---:|---:|---:|
| 头 1 | 0.4011 | 0.4011 | 0.1978 |
| 头 2 | 0.2483 | 0.5035 | 0.2483 |

每个头用自己的权重对自己的 Value 加权求和：

$$
\begin{aligned}
o_2^{(1)}
&=0.4011[1,0]+0.4011[1,0]+0.1978[0,1]\\
&\approx[0.8022,\ 0.1978]
\end{aligned}
$$

$$
\begin{aligned}
o_2^{(2)}
&=0.2483[1,0]+0.5035[0,1]+0.2483[1,0]\\
&\approx[0.4965,\ 0.5035]
\end{aligned}
$$

将两个头拼接：

$$
z_2=[0.8022,\ 0.1978,\ 0.4965,\ 0.5035]
$$

对全部 token 计算，可以得到：

$$
Z\approx
\begin{bmatrix}
0.8022&0.1978&0.8022&0.1978\\
0.8022&0.1978&0.4965&0.5035\\
0.4965&0.5035&0.8022&0.1978
\end{bmatrix}
$$

最后做输出投影 $Y=ZW_O$。选择一个便于观察跨头混合的示例矩阵：

$$
W_O=
\begin{bmatrix}
0.5&0&0.5&0\\
0&0.5&0&0.5\\
0.5&0&-0.5&0\\
0&0.5&0&-0.5
\end{bmatrix}
$$

它将前后两组 2 维向量转为“平均值和差值”，得到：

$$
Y\approx
\begin{bmatrix}
0.8022&0.1978&0&0\\
0.6494&0.3506&0.1529&-0.1529\\
0.6494&0.3506&-0.1529&0.1529
\end{bmatrix}
$$

最终仍为 `(3, 4)`。上述数值经过四舍五入；真实模型的 $W_O$ 同样通过训练学习。

批处理实现中的形状变化为：

```text
输入                         (B, N, d_model)
Q、K、V 投影                 (B, N, d_model)
拆头、转置                   (B, h, N, d_k)
每头注意力分数               (B, h, N, N)
每头加权汇总                 (B, h, N, d_v)
转置、拼接                   (B, N, d_model)
输出投影                     (B, N, d_model)
```

这里使用标准设置 `h × d_k = h × d_v = d_model`。如果用于 Decoder-only，需要在 Softmax 前将未来位置分数设为负无穷；上述数值例子没有加入因果 Mask。

### 24. 每个头只在头内计算，为什么多头还有优势？

这里的“更多信息”应理解为：**允许不同子空间使用不同的注意力分布，从同一个输入中保留不同方面的关系。**

第一，每个头都可以利用完整输入特征。将大投影矩阵按列拆开：

$$
W_Q=[W_Q^{(1)}\mid W_Q^{(2)}]
$$

$$
Q^{(1)}=XW_Q^{(1)},\qquad Q^{(2)}=XW_Q^{(2)}
$$

当 $X:(N,4)$、$W_Q^{(1)}:(4,2)$ 时，头 1 的两维来自完整四维输入。例如可以学成：

$$
q^{(1)}=[x_1+2x_3,\ x_2-x_4]
$$

上一节的单位矩阵让拆头看起来像直接取原输入的前两维、后两维；那是手算特例。一般投影会组合完整输入，K、V 也是如此。

第二，对同一个查询位置，单头只有一组 token 权重：

$$
o_i=\sum_j a_{ij}v_j
$$

该输出向量的各个特征通道使用同一组标量权重 $a_{ij}$。多头则允许：

$$
o_i^{(r)}=\sum_j a_{ij}^{(r)}v_j^{(r)}
$$

不同头的 $a_{ij}^{(r)}$ 可以不同。

例如处理“`小明 在 北京 工作`”中的“工作”，两个头可能形成以下分布：

| 当前查询为“工作” | 小明 | 在 | 北京 | 工作 |
|---|---:|---:|---:|---:|
| 头 1 | 0.85 | 0.05 | 0.05 | 0.05 |
| 头 2 | 0.05 | 0.05 | 0.85 | 0.05 |

头 1 偏向提取“谁在工作”，头 2 偏向提取“在哪里工作”。这是帮助理解的示意，实际分工由训练形成，并不保证每个头都有清晰的人类语义。

第三，各头结果经拼接和 $W_O$ 混合，下一层又可以基于完整输出进行投影，所以信息不会永远隔离在不同头中。

**多头不保证一定优于单头，也不是头越多越好。** 固定总隐藏维度时，增加头数会减小每个头的维度，一些头也可能冗余。它提供的是更灵活的多组信息选择和聚合方式。

## GPU 架构与计算量

### 25. MMA 的 4096 FLOPs 是怎样算出来的？

MMA（Matrix Multiply-Accumulate）执行矩阵乘累加：

$$
D=AB+C
$$

对 PTX 指令：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

`m16n8k16` 表示 $m=16,n=8,k=16$，矩阵形状为：

| 矩阵 | 通用形状 | 本例形状 |
|---|---|---|
| A | $m\times k$ | $16\times16$ |
| B | $k\times n$ | $16\times8$ |
| C、D | $m\times n$ | $16\times8$ |

一个输出元素：

$$
D_{ij}=C_{ij}+\sum_{\ell=0}^{15}A_{i\ell}B_{\ell j}
$$

按数学运算计数，可以理解为：

```cpp
float acc = C[i][j];
for (int k = 0; k < 16; ++k) {
    acc += A[i][k] * B[k][j];
}
D[i][j] = acc;
```

每个输出元素执行 16 次乘累加。一次标量乘累加按一次乘法、一次加法计，即 2 FLOPs：

$$
\text{每个输出元素}=16\times2=32\ \text{FLOPs}
$$

D 有 $16\times8=128$ 个元素，因此：

$$
\text{总计算量}
=128\times32
=2\times16\times8\times16
=4096\ \text{FLOPs}
$$

一般公式为：

$$
\boxed{\text{MMA 计算量}=2mnk}
$$

要点：

- 加 C 已包含在内：16 个乘积求和需要 15 次加法，再加 C 共 16 次加法。
- 硬件即使融合乘加，运算量仍按 2 FLOPs 计数。
- 4096 FLOPs 是整个 Warp 协作完成这次 MMA 的总量，不再乘以 32。
- 指令形状不能直接推出单个 Tensor Core 每周期吞吐量，吞吐量和延迟要查具体架构。
- `FLOPs` 表示运算量；`FLOP/s` 或规格表中的 `FLOPS` 表示每秒运算速度。

### 26. GPU 的 SM 相当于 CPU 中的什么？

按芯片内部执行单元的层级，可以把 SM 粗略类比为 CPU Core，但两者内部结构、线程规模和性能不能一一对应。

```text
CPU 芯片
  └── 多个 CPU Core
        └── ALU、SIMD 等运算单元

GPU 芯片
  └── 多个 SM
        └── CUDA Core、Tensor Core 等运算单元
```

CUDA Core 更接近 CPU 核心内部的算术执行通道，而不是一个完整 CPU 核心。

| 对比 | CPU Core | GPU SM |
|---|---|---|
| 优化重点 | 单个或少量线程的低延迟 | 大量线程的整体吞吐量 |
| 执行组织 | 常见有乱序执行、分支预测、缓存等 | Warp 调度器选择就绪 Warp 执行 |
| 线程规模 | 常见支持 1～2 个硬件线程，也有其他设计 | 可驻留数百乃至上千线程，取决于架构和资源 |
| 等待数据时 | 通过缓存、预取、乱序执行等隐藏延迟 | 调度其他就绪 Warp 继续计算 |

结合原文的 H100，一个 SM 中包含 128 个 FP32 CUDA Core、4 个 Tensor Core，以及调度器、寄存器、共享内存等。

**驻留不等于同时执行。** 一个 SM 可以保存很多 Warp 的状态，但每周期能发射、执行多少工作受调度器、执行单元和数据依赖限制。

## 易混淆概念速查

| 概念 | 应该怎样理解 |
|---|---|
| `xs = ...` 与 `xs[:] = ...` | 前者重新绑定名字；后者修改当前列表内容 |
| `append` 与 `copy` | 前者修改原列表；后者返回浅拷贝 |
| 生成器函数调用与函数体执行 | 调用先返回生成器，迭代时才执行函数体 |
| 装饰与调用 | 装饰时创建包装；调用时执行包装逻辑 |
| 并发与并行 | 并发是多个任务交替推进；并行是同一时刻执行 |
| GIL 与业务正确性 | GIL 不替代业务层的锁、事务和同步协议 |
| 通信与同步 | 通信传递信息；同步约束访问和执行顺序 |
| `weak_ptr::lock()` 与互斥锁 | 前者尝试取得强引用，不是给资源加互斥锁 |
| `std::move` 与实际移动 | 前者转换表达式类别；实际资源转移由类型实现 |
| 成员函数 `const` 与线程安全 | 只读接口约束不自动保证并发安全 |
| CMake 与编译器 | CMake 组织、生成构建规则；编译器执行编译 |
| `free` 与 `available` | 完全空闲内存与估计可供应用使用的内存 |
| `ps` 与 `top` | 一次快照与持续监视，CPU 统计时间范围也不同 |
| Encoder/Decoder 与 Prefill/Decode | 模型结构与推理执行阶段 |
| $d_{\text{model}}$ 与 $d_k$ | 总隐藏维度与每次注意力点积的维度 |
| 注意力缩放 | 除以 $\sqrt{d_k}$ |
| 多头与更多输入信息 | 多头提供多组关注与聚合方式，不增加外部输入 |
| CUDA Core 与 SM | 运算单元与包含运算、调度、存储资源的执行簇 |
| MMA 计算量与硬件吞吐量 | 一次操作做多少运算，与每秒/每周期能做多少不同 |

## 自我检验清单

- [ ] 能画出 `mutate` 与 `rebind` 前后名字和对象的指向。
- [ ] 能解释 `xs[:]` 读取和切片赋值的区别。
- [ ] 能说明为什么生成器交出批次后用 `batch = []`。
- [ ] 能区分装饰器执行时机与包装函数执行时机。
- [ ] 能解释进程池为什么适合“输入输出少、计算量大”的任务。
- [ ] 能复现一个单线程协程中的更新丢失问题。
- [ ] 能按共享资源选择线程锁、异步锁、进程锁或数据库事务。
- [ ] 能解释 RAII、强引用计数与循环引用。
- [ ] 能区分拷贝构造、移动构造、拷贝赋值和移动赋值。
- [ ] 能解释三种 `const` 位置分别限制什么。
- [ ] 能分别举出词法、语法、语义错误。
- [ ] 能使用 CMake 配置并构建一个小项目。
- [ ] 能用 `ps` 找进程、用 `top` 观察变化、用 `free` 看可用主机内存。
- [ ] 能说明 Self-Attention 和 Cross-Attention 的 QKV 来源。
- [ ] 能在 `d_model = 512, h = 8` 时算出每头维度和缩放因子。
- [ ] 能手算两头注意力，并解释不同头的权重为什么可以不同。
- [ ] 能说明头内独立计算与头间输出混合的关系。
- [ ] 能从输出元素数量推导 MMA 的 $2mnk$。
- [ ] 能区分 SM、CUDA Core、Warp，以及运算量和吞吐量。

## 参考原文

本笔记的基础阅读材料：

- [第 1 章：编程语言基础](../guides/模块一-前置知识/第1章-编程语言基础.md)
- [3.3 Self-Attention 机制深入理解](<../guides/模块一-前置知识/transformer/3.3 Self-Attention机制深入理解.md>)
- [GPU 基础知识](../guides/模块一-前置知识/gpu/gpu-basics.md)，重点为 4.1 MMA 工作原理。

本文按提问主题归纳，不替代原章节的完整学习路线。复习时先独立回答问题，再检查代码、维度和计算过程。
