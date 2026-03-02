# CLI 命令

## 1. CMake + Make（CUDA 版本）

```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON
make -j
```

## 2. 运行 MNIST（CUDA，单卡示例）

```bash
cd /home/maxxx_1/2026_winter/mx-InfiniTrain/build
CUDA_VISIBLE_DEVICES=0 ./mnist --dataset ~/datasets/mnist --device cuda --num_epoch 20 --bs 256 --lr 0.01
```

## 3. LLaMA3 单卡训练
> ⚠️ 注意：LLaMA3 权重较大，单 GPU（如 24 GB 显存）在默认配置下容易 OOM。如果看到类似
> `CUDA Error: out of memory`，请减小 `--batch_size`、`--total_batch_size` 或
> `--sequence_length`，或换用更小的模型/开启 BF16、混合精度。
```bash
cd /home/maxxx_1/2026_winter/mx-InfiniTrain/build

./llama3 \
  --device cuda \
  --input_bin /data/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin \
  --tokenizer_bin /data/shared/InfiniTrain-dev/data/llmc/llama3/llama3_tokenizer.bin \
  --llmc_filepath /data/shared/InfiniTrain-dev/data/llmc/llama3/llama3.2_1B_fp32.bin \
  --sequence_length 64 \
  --batch_size 1 \
  --total_batch_size 64 \
  --num_iteration 10
```

## 4. GPT2 单卡训练

```bash
cd /home/maxxx_1/2026_winter/mx-InfiniTrain/build

./gpt2 \
  --device cuda \
  --input_bin /data/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin \
  --tokenizer_bin /data/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_tokenizer.bin \
  --llmc_filepath /data/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_124M.bin \
  --sequence_length 64 \
  --batch_size 4 \
  --total_batch_size 256 \
  --num_iteration 10
```