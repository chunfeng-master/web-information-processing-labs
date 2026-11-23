# Web 信息处理与应用课程实验

本仓库整理课程实验代码与报告。

## 实验 1：信息检索

内容包括中文分词、位置倒排索引、索引压缩、布尔与短语检索、TF-IDF 和 BGE 文档表征。

`lab1/` 来源于小组仓库 [yunleeeee/web_lab1](https://github.com/yunleeeee/web_lab1)，保留原代码、说明和报告中的成员署名。

- 同步日期：2026-11-23
- 来源提交：[`57a1879c1e3791e7357f75d2e3aa26e06c1b3821`](https://github.com/yunleeeee/web_lab1/tree/57a1879c1e3791e7357f75d2e3aa26e06c1b3821)
- [详细来源记录](./lab1/SOURCE.md)
- [环境配置与运行步骤](./lab1/README.md)
- [实验报告](./lab1/docs/lab1_report.md)
- [实验要求](./lab1/docs/实验1(1).pdf)

原先的实验 1 实现已由小组版本替换，历史版本仍可通过 Git 提交记录查阅。

### 运行方法

进入 `lab1/`，按其 README 中对应操作系统的步骤配置环境、安装依赖、准备 Qilin-IR-50K 数据集和下载 BGE 模型，再依次运行 `prepare`、`lexical`、`semantic` 阶段，或运行 `all`。

数据集、模型、虚拟环境和实验输出不随源仓库提供，也不提交到本仓库。结果需要按小组版本重新生成，不能沿用旧实现的结果。

## 实验 2：电影知识图谱链接预测

从 Freebase 中抽取 1,600 部电影及其相关实体，得到包含 4,172 个实体、35 种关系和 8,639 条三元组的子图。在助教框架中补全 TransE 和尾实体负采样，再用电影属性和相似电影的信息改进排序。

| 排序方法 | 测试 MRR | Hits@5 | Hits@10 | 平均名次 |
| --- | ---: | ---: | ---: | ---: |
| TransE 距离 | 0.4229 | 52.84% | 60.95% | 242.89 |
| TransE 距离与电影属性联合排序 | 0.5635 | 71.03% | 78.91% | 48.17 |

两种方法使用相同的 863 条测试查询。[实验报告](./lab2/docs/实验二实验报告.md) · [实验要求](./lab2/docs/实验2.pdf) · [助教框架与实现](./lab2/framework/)

### 运行方法

仓库已包含[训练、验证和测试数据](./lab2/data/freebase/)及[实体和关系编号表](./lab2/data/metadata/)。从 `lab2/` 目录运行 `python -B framework/main_kg.py` 即可开始训练。依赖版本、加载最佳权重和复现两组测试结果的命令见[报告中的运行方法](./lab2/docs/实验二实验报告.md#运行方法)。

如需重新抽取子图，将课程原始文件放到 `lab2/data/raw/freebase_douban.gz`，再从 `lab2/` 运行 `python -B framework/loader_kg.py`。
