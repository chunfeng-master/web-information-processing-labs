# Web 信息处理与应用课程实验

本仓库用于整理“Web 信息处理与应用”课程实验，包括实验代码、实验报告、测试程序和关键实验结果。

## 实验列表

| 实验 | 主要内容 | 状态 |
|---|---|---|
| [实验 1：信息检索](./lab1/) | 中文分词与规范化、倒排索引、位置索引、索引压缩、布尔检索、TF-IDF 与文档表征检索 | 已完成 |

## 实验 1 完成内容

实验 1 使用 Qilin-IR-50K 数据集，完成以下必做内容：

1. 使用双向最大匹配和统计方法进行中文分词，并比较两种方法的结果。
2. 对文本进行大小写、数字、符号和停用词规范化处理。
3. 构建包含文档编号、词频和词项位置的倒排索引。
4. 使用位置索引支持短语检索，并使用跳表优化倒排表求交。
5. 实现按块存储、前端编码以及 docID 间距编码与可变字节码。
6. 实现复杂布尔检索、TF-IDF 向量空间检索和文档表征检索。
7. 比较不同索引压缩方案、检索顺序、跳表步长及文档表征方法。

详细内容参见：

- [实验 1 使用说明](./lab1/README.md)
- [实验 1 报告](./lab1/实验报告.md)
- [实验结果说明](./lab1/outputs/README.md)
- [机器可读实验结果](./lab1/outputs/results.json)

## 仓库结构

```text
LAB/
├── README.md
├── .gitignore
└── lab1/
    ├── src/                    # 分词、索引、压缩和检索核心实现
    ├── tests/                  # 核心算法单元测试
    ├── outputs/                # 结果说明与关键统计结果
    ├── run_experiment.py       # 完整实验入口
    ├── verify_completion.py    # 必做内容与结果一致性检查
    ├── requirements.txt        # Python 依赖
    ├── README.md               # 实验 1 运行说明
    └── 实验报告.md             # 实验报告
```

## 运行实验 1

建议使用 Python 3.10 或更高版本。在仓库根目录执行：

```powershell
cd lab1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
python verify_completion.py
```

首次运行语义检索时需要下载 `paraphrase-multilingual-MiniLM-L12-v2` 模型。若只检查非语义部分，可执行：

```powershell
python run_experiment.py --skip-semantic
```

## 数据与大型结果文件

Qilin-IR-50K 原始数据集和可由程序重新生成的大型文件未上传到普通 Git 仓库，包括：

```text
lab1/Qilin-IR-50K/
lab1/Qilin-IR-50K.rar
lab1/outputs/*.sqlite
lab1/outputs/*.jsonl
```

运行实验前，请将数据集解压到：

```text
lab1/Qilin-IR-50K/
```

大型文件未纳入版本管理的原因是它们体积较大，并且部分单文件超过 GitHub 普通 Git 的 100 MB 限制。仓库保留 `outputs/results.json`，用于展示完整实验的关键统计、查询案例和对比结果。

## 说明

- 本仓库目前只包含实验文档要求中的必做内容，选做部分未实现。
- 实验程序固定随机种子为 `2026`，便于复现实验结果。
- 计时结果与运行设备、磁盘缓存和当前系统负载有关。
- 点击数据仅作为相关性的辅助指标，未点击不代表不相关。
