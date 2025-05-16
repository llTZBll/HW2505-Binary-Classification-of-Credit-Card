# 信用卡推荐预测任务（二分类）

## 项目概述

本项目是数据统计与分析基础的大作业，是一个二分类任务，旨在预测客户是否有兴趣开办信用卡（Is_Lead）。项目采用LightGBM模型，并使用贝叶斯优化方法寻找最佳参数组合，以提高预测准确性。

## 方法说明

1. 预测目标`Is_Lead`不平衡，为什么不做采样操作？
   + AUC对类别不平衡不敏感
   + `Is_Lead`不平衡特点不显著，作为Boosting方法具有一定的鲁棒性
   + 已经实验证明此处使用采样操作不会提升在原数据上的AUC
2. 一些参数
   + 优化器默认迭代次数：50次
   + 已经训练的模型和推理结果的迭代次数：500次

## 项目结构

```
project/
├─ data/                   # 数据存储文件夹
│  └─ train.csv            # 原始训练数据（保留原始文件）
│  └─ test.csv             # 测试数据（无预测项）
├─ src/                    # 核心代码文件夹
│  ├─ utils/               # 工具函数模块
│  │  ├─ preprocessing.py  # 数据预处理相关函数
│  │  └─ evaluation.py     # 模型评估相关函数
│  ├─ models/              # 模型相关代码
│  │  └─ lgb_optimizer.py  # LightGBM优化器类
│  ├─ config.py            # 全局配置文件
│  ├─ train.py             # 主训练脚本
│  ├─ predict.py           # 主运行脚本
│  └─ requirements.txt     # 依赖环境文件
├─ results/                # 结果输出文件夹
│  ├─ models/              # 训练好的模型文件
│  ├─ logs/                # 日志文件
│  ├─ plots/               # 图表文件
│  └─ reports/             # 报告文件（如优化结果CSV）
└─ README.md               # 项目说明文档
```

## 环境配置

1. 创建虚拟环境（可选但推荐）


2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 数据准备

将原始数据集`train.csv`放入`data/`目录下。

## 运行步骤

1. 确保数据已正确放置在`data/`目录下。

2. 执行训练程序（默认将使用优化器迭代超参数50次）：

   ```bash
   python src/train.py
   ```

3. 查看`train.py`结果：

   - 模型优化过程会保存在`results/reports/lgb_bayesian_optimization.csv`
   - 训练好的模型会保存在`results/models/final_model.txt`
   - 日志文件会保存在`results/logs/training.log`
   - 图表（优化过程、ROC曲线、特征重要性）会保存在`results/plots/`目录下

4. 执行推理程序：

   ```bash
   python src/predict.py
   ```

5. 查看推理结果：

   - 预测结果存放到`result/reports/predictions.csv`

## 代码说明

- `preprocessing.py`: 包含数据加载、缺失值处理和特征编码等预处理函数。
- `evaluation.py`: 包含模型评估和可视化相关函数。
- `lgb_optimizer.py`: 实现LightGBM模型的贝叶斯优化。
- `config.py`: 集中管理项目配置参数和路径。
- `train.py`: 模型训练主入口，协调数据处理、模型训练和评估流程。
- `predict.py`：模型推理主入口，用于从`test.csv`生成预测结果存放到`result/reports/predictions.csv`

## 结果分析

运行完成后，可以通过以下方式分析结果：

1. 查看`results/plots/optimization_progress.png`了解优化过程中AUC的变化。
2. 查看`results/plots/feature_importance.png`了解哪些特征对预测最有影响。
3. 查看`results/reports/feature_importance.csv`获取详细的特征重要性数值。
4. 查看`results/logs/training.log`获取完整的训练过程记录。  

## 更进一步

1. 可以尝试使用其他的模型完成该任务；
2. 可以增大优化器迭代次数和超参数搜索范围；
3. 可以使用其他的缺失值填补方式；
4. 可以对原始数据特征进行更详细的分析。
