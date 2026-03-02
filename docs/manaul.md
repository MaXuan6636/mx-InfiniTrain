# MNIST 代码执行链路梳理

## 1）入口：`example/mnist/main.cc`

运行示例：

```bash
./mnist --dataset ... --device cuda --num_epoch 10 --bs 256 --lr 0.01
```

`main.cc` 的启动阶段主要做三件事：

- 用 `gflags` 解析命令行参数；
- 构建训练/测试数据集与 `DataLoader`；
- 构建模型、损失函数、优化器。

---

## 2）数据集：`example/mnist/dataset.cc`

`MNISTDataset` 的职责：

- 读取 4 个 MNIST idx 文件（`train/t10k` 的 `images + labels`）；
- 把图像从 `uint8` 转成 `float32` 并做 `/255.0` 归一化；
- 实现 `operator[](idx)`，返回第 `idx` 个样本的 `(image, label)` 两个 tensor。

说明：当前实现里已经按 tensor 当前状态动态计算样本偏移，避免了 dtype 变化后的错位读取问题。

---

## 3）DataLoader：`infini_train/src/dataloader.cc`

`DataLoader` 每次会拿一批样本，然后把单样本 tensor 堆叠成 batch tensor。

关键细节：

- 堆叠时会把每个样本展平成一维再拼接；
- 所以图像最终形状是 `(bs, 784)`（而不是 `(bs, 28, 28)`）。

---

## 4）模型：`example/mnist/net.cc`

网络结构非常简单：

- `Linear(784, 30)`
- `Sigmoid`
- `Linear(30, 10)`

输入 `(bs, 784)`，输出 `(bs, 10)` 的 logits。

---

## 5）训练循环：每个 batch 的执行顺序

在 `main.cc` 训练循环里，每个 batch 执行：

1. 把 `image/label` 拷到目标设备（CPU 或 CUDA）；
2. 前向计算：`network.Forward({new_image})`；
3. 清梯度：`optimizer.ZeroGrad()`；
4. 计算损失：`loss_fn.Forward({outputs[0], new_label})`；
5. 反向传播：`loss[0]->Backward()`；
6. 参数更新：`optimizer.Step()`。

每个 epoch 末尾会打印：

- `train loss`
- `lr`
- `epoch 耗时`
- `samples/s`

---

## 6）评估循环：准确率如何计算

训练结束后，在测试集上执行：

- 前向得到 logits；
- 对每个样本取 `argmax(logits)` 作为预测类别；
- 与真实标签比较，统计 `correct/total`；
- 输出 `Accuracy` 和 `AverageLoss`。

---

## 7）一张脑内流程图

命令行参数  
-> `MNISTDataset`（读文件 + 归一化）  
-> `DataLoader`（按 batch 堆叠）  
-> `MNIST` 网络前向  
-> `CrossEntropy`  
-> `Backward`  
-> `SGD Step`  
-> 每 epoch 打印训练指标  
-> 最后输出测试集准确率
