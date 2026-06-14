# UHR-Net

面向医学图像分割的 **不确定性感知超图细化网络**  
**UHR-Net: An Uncertainty-Aware Hypergraph Refinement Network for Medical Image Segmentation**

<p align="center">
  <a href="https://arxiv.org/abs/2604.28095"><img src="https://img.shields.io/badge/arXiv-2604.28095-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/CUGfreshman/UHR-Net"><img src="https://img.shields.io/badge/Code-UHR--Net-blue.svg" alt="Code"></a>
  <img src="https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-green.svg" alt="Task">
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <img src="assets/readme/framework.png" width="95%" alt="UHR-Net framework">
</p>

## 概览

**UHR-Net** 将实例级对比预训练与不确定性引导的结构化细化结合起来，旨在缓解医学病灶分割中的小病灶线索稀释、类病灶背景干扰以及模糊区域预测不稳定问题。

### 不确定性导向的实例级对比预训练策略 (UO-IC)

- UO-IC 是一种实例级对比预训练策略，结合几何约束的 copy-paste 增强和类病灶背景困难负样本挖掘，以缓解小病灶线索稀释和类病灶背景干扰。
- 正样本对由几何约束的 copy-paste 构造：将原始病灶 Lesion A 随机缩放并粘贴为 Lesion B，通过掩码平均池化分别得到原始病灶特征 `z_A` 和粘贴病灶特征 `z_B`；负样本来自预测前景概率较高的类病灶背景区域，通过加权掩码平均池化得到类病灶背景特征 `z_bg`，并作为 batch 内困难负样本。
- 基于上述正、负样本，UO-IC 使用 InfoNCE 进行实例级对比优化，拉近原始病灶特征 `z_A` 与其缩放粘贴副本特征 `z_B`，同时推远 `z_A` 与类病灶背景特征 `z_bg`，从而缓解小病灶线索稀释和类病灶背景干扰。

### 不确定性引导的超图细化模块 (UGHR)

- **UGHR 块** 被嵌入多尺度解码路径，用于在分割阶段细化解码器特征；每个 UGHR 块接收当前尺度的融合特征和粗分割概率图，并输出细化后的特征。
- UGHR 从粗分割概率图计算基于熵的不确定性图，并在归一化前用该不确定性调制节点-超边参与 logits，使高不确定性位置获得更大的归一化参与权重，从而在后续超图消息传递中更有效地聚合这些区域的上下文信息。
- UGHR 将超边原型划分为前景组和背景组，并根据粗分割先验提取前景/背景上下文生成动态原型；这种前景/背景条件化的超边原型用于解耦高阶交互，减少边界干扰并增强不确定区域的细化效果。

## 主要结果

下表展示了 UHR-Net 在 **Kvasir-Sessile、Kvasir-SEG 和 GlaS** 数据集上的主要定量结果。

<p align="center">
  <img src="assets/readme/main_results_table1.png" width="100%" alt="Main results on Kvasir-Sessile, Kvasir-SEG and GlaS">
</p>
## 快速开始

### 1. 环境配置

创建 conda 环境并安装依赖：

```bash
git clone https://github.com/CUGfreshman/UHR-Net.git
cd UHR-Net

conda create -n uhrnet python=3.10 -y
conda activate uhrnet

# 请根据本机 CUDA 版本安装匹配的 PyTorch 版本
pip install torch torchvision
pip install opencv-python albumentations numpy scipy scikit-image scikit-learn tqdm
```

如果默认 `pip install torch torchvision` 没有安装到你需要的 CUDA 版本，请参考 PyTorch 官方安装命令重新安装匹配本机 CUDA 的 PyTorch。

### 2. 数据准备

默认脚本从 `--data-root` 下读取数据集。数据集目录名不作强制要求，只需要和运行命令中的 `--dataset` 参数保持一致即可。下面以 `Kvasir-SEG` 为例：

```text
data/
└── Kvasir-SEG/
    ├── images/
    ├── masks/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── preprocessed_metadata.pkl   # 由下面的 UO-IC 预处理脚本生成
```

UO-IC 训练前需要构建 copy-paste 所需的实例与距离图元数据，请运行对应预处理脚本。默认输出为数据集目录下的 `preprocessed_metadata.pkl`。

```bash
# Kvasir-SEG
python scripts/preprocess_kvasir.py \
  --data-root data \
  --dataset Kvasir-SEG

# GlaS
python scripts/preprocess_glas.py \
  --data-root data \
  --dataset Glas

# ISIC-2016
python scripts/preprocess_isic2016.py \
  --data-root data \
  --dataset ISIC-2016
```

如果你已经有预处理好的元数据，也可以在 stage-1 训练时通过 `--metadata-path` 指定路径。

### 3. Stage-1 UO-IC 预训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_uoic.py \
  --data-root data \
  --dataset Kvasir-SEG \
  --metadata-path data/Kvasir-SEG/preprocessed_metadata.pkl \
  --output-root run_files \
  --epochs 300 \
  --batch-size 16
```

训练日志和 checkpoint 默认保存到：

```text
run_files/<dataset>/stage1_<dataset>_<timestamp>/
```

其中 `checkpoint.pth` 用于下一阶段的 backbone 初始化。

### 4. UHR-Net 端到端训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_uhrnet.py \
  --data-root data \
  --dataset Kvasir-SEG \
  --pretrained-backbone path/to/stage1/checkpoint.pth \
  --output-root run_files \
  --epochs 300 \
  --batch-size 24
```

`--pretrained-backbone` 是可选参数；如果不提供，模型将从随机初始化的 UHR-Net 开始训练。若需要从完整 UHR-Net checkpoint 继续训练，请使用 `--resume path/to/checkpoint.pth`。

训练输出默认保存到：

```text
run_files/<dataset>/<dataset>_<timestamp>/
```

### 5. 测试

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test_uhrnet.py \
  --data-root data \
  --dataset Kvasir-SEG \
  --checkpoint path/to/uhrnet/checkpoint.pth \
  --split val
```

`--checkpoint` 为必填参数。`--split` 支持 `train`、`val` 和 `test`；使用 `test` 时，数据集目录中需要提供 `test.txt`。

测试脚本默认在 checkpoint 所在目录下写入日志文件：

```text
test_<split>.log
```

## 仓库结构

```text
UHR-Net/
├── README.md
├── README_CN.md
├── assets/
│   └── readme/                 # README 图片资源
├── data/
│   ├── io.py                    # 数据列表读取与路径解析
│   ├── segmentation_dataset.py   # 端到端分割训练数据集
│   └── uoic_dataset.py           # UO-IC 预训练数据集与 copy-paste 构造
├── engine/
│   ├── uhrnet_engine.py          # UHR-Net 训练/验证逻辑
│   └── uoic_engine.py            # UO-IC 预训练逻辑
├── models/
│   ├── resnet.py                 # ResNet backbone
│   ├── ughr.py                   # UGHR block
│   ├── uhr_net.py                # UHR-Net 主网络
│   └── uoic_pretrain.py          # UO-IC 预训练网络
├── scripts/
│   ├── preprocess_glas.py        # GlaS 元数据预处理
│   ├── preprocess_isic2016.py    # ISIC-2016 元数据预处理
│   ├── preprocess_kvasir.py      # Kvasir 元数据预处理
│   ├── train_uoic.py             # UO-IC 预训练入口
│   ├── train_uhrnet.py           # UHR-Net 端到端训练入口
│   └── test_uhrnet.py            # 测试入口，输出 IoU 日志
└── utils/
    ├── metrics.py                # 损失函数与评价指标
    └── utils.py                  # 日志、随机种子、shuffle 等工具函数
```

## 问题与支持

如果您在数据集准备、训练、验证或测试过程中遇到任何困难，欢迎及时与我们联系，我们会尽力提供帮助。

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{cheng2026uhrnet,
  title        = {UHR-Net: An Uncertainty-Aware Hypergraph Refinement Network for Medical Image Segmentation},
  author       = {Cheng, Shuokun and Shi, Jinghao and Sun, Kun},
  year         = {2026},
  eprint       = {2604.28095},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2604.28095}
}
```
