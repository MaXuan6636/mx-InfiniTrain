# 仓库文件结构（关键部分）  # 说明：tree 风格 + 行末注释
#
# 注意  # 本结构图聚焦关键目录/入口文件，不追求列出每一个源码文件
#
```bash
mx-InfiniTrain/  # 仓库根目录（C++ 大模型训练框架）
├── .clang-format  # C++/CUDA 代码格式化规则（clang-format）
├── .gitignore  # Git 忽略规则
├── .gitmodules  # Git 子模块声明（third_party 等可能以 submodule 形式管理）
├── .github/  # GitHub 配置（CI 等）
│   └── workflows/  # GitHub Actions 工作流
│       └── format-check.yaml  # CI：格式检查（如 black/clang-format 等）
├── CMakeLists.txt  # 顶层构建脚本：选项、依赖、库与可执行程序定义
├── LICENSE  # 开源许可证
├── README.md  # 项目介绍与快速开始（编译/训练/并行示例）
├── cmake/  # CMake 辅助模块目录
│   └── FindNCCL.cmake  # 查找/链接 NCCL 的 CMake 模块
├── docs/  # 项目文档
│   ├── hook_mechanism.md  # Hook 机制设计说明（Module / Function hooks）
│   ├── precision_checker_guide.md  # 精度检查工具使用指南（NaN/Inf、MD5、NPY）
│   └── repo_structure.md  # 本文件：仓库结构图与模块说明
├── example/  # 示例工程：每个子目录编译成一个独立可执行程序
│   ├── common/  # LLM 示例共享组件（数据集/Tokenizer/工具函数）
│   ├── gpt2/  # GPT-2 训练示例（SFT/并行/混精/精度检查）
│   ├── llama3/  # LLaMA3 训练示例（SFT/多维并行/混精/精度检查）
│   └── mnist/  # MNIST 入门示例（最小训练闭环，便于验证框架）
├── infini_train/  # 框架核心实现（头文件 + 源码）
│   ├── include/  # 对外暴露的 C++ API（类似“头文件库”入口）
│   │   ├── autocast.h  # Autocast 混精上下文（如 BF16 compute + FP32 accum）
│   │   ├── dataloader.h  # DataLoader 接口与实现声明
│   │   ├── dataset.h  # Dataset 抽象与数据集相关声明
│   │   ├── datatype.h  # 数据类型定义（float32/bfloat16 等）
│   │   ├── device.h  # 设备抽象（CPU/CUDA）
│   │   ├── dispatcher.h  # Kernel 注册/调度相关接口
│   │   ├── optimizer.h  # Optimizer 接口（SGD/Adam 及并行优化器封装）
│   │   ├── profiler.h  # 内置 profiler 接口（PROFILE_MODE 下输出 report/records）
│   │   ├── tensor.h  # Tensor 核心抽象（数据、shape、autograd 关联等）
│   │   ├── autograd/  # 自动微分相关头文件（Function、Hook、各类算子梯度）
│   │   ├── common/  # 通用基础设施（如统一 hook handle 等）
│   │   ├── core/  # 核心运行时抽象（stream、device guard、blas handle）
│   │   ├── nn/  # 神经网络层与并行训练接口（modules/parallel/functional/init）
│   │   └── utils/  # 工具：precision checker、全局 hook registry 等
│   └── src/  # 框架源码实现
│       ├── autograd/  # Autograd 实现（Function 执行、反向传播、hook 调用等）
│       ├── core/  # 运行时实现（CPU/CUDA guard、stream、blas handle 等）
│       ├── kernels/  # CPU/CUDA kernels（算子具体实现）
│       │   ├── cpu/  # CPU 算子实现（.cc）
│       │   └── cuda/  # CUDA 算子实现（.cu，需 USE_CUDA=ON）
│       ├── nn/  # NN 层实现（modules/）与并行实现（parallel/）
│       │   ├── modules/  # Module 子类实现（Linear/LN/Loss/Container 等）
│       │   └── parallel/  # 并行训练实现（DP/DDP/TP/PP/SP、process group 等）
│       ├── utils/  # 工具实现（precision checker、global hook registry 等）
│       ├── dataloader.cc  # DataLoader 实现
│       ├── device.cc  # Device 实现
│       ├── optimizer.cc  # Optimizer 实现
│       ├── profiler.cc  # Profiler 实现（PROFILE_MODE）
│       └── tensor.cc  # Tensor 实现
├── scripts/  # 运行与分析脚本（多为 Python/Bash）
│   ├── run_models_and_profile.bash  # 批量 build+run+收集日志（可选 profiling）
│   ├── test_config.json  # run_models_and_profile.bash 的测试配置（builds/tests/变量）
│   ├── compare_loss.py  # 对比两次运行日志的 loss（用于回归）
│   ├── compare_tps.py  # 对比两次运行日志的 tok/s 吞吐（用于回归）
│   ├── format.py  # 代码格式化脚本（仓库风格工具链的一部分）
│   ├── write_to_feishu_sheet.py  # 将训练指标写入飞书表格（自动化汇报）
│   └── precision_check/  # 精度检查的离线分析脚本目录
│       └── precision_compare.py  # 对比两次精度检查导出的 NPY（atol/rtol）
├── test/  # 测试目录
│   └── hook/  # Hook / precision checker 相关测试
│       ├── test_hook.cc  # Hook 机制功能测试
│       └── test_precision_check.cc  # 精度检查功能测试（simple/md5/npy 等）
├── tools/  # 工具类可执行程序
│   └── infini_run/  # 分布式启动器（fork 多进程 + 注入 env）
│       ├── CMakeLists.txt  # 构建 infini_run 工具
│       └── infini_run.cc  # 启动器实现（设置 NNODES/RANK/MASTER_ADDR 等）
└── third_party/  # 第三方依赖（当前仓库中已包含/或以 submodule 形式管理）
    ├── eigen/  # Eigen（线性代数库）
    ├── gflags/  # gflags（命令行参数解析）
    └── glog/  # glog（日志库）

```