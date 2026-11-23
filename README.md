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

## 使用说明

进入 `lab1/`，按其 README 中对应操作系统的步骤配置环境、安装依赖、准备 Qilin-IR-50K 数据集和下载 BGE 模型，再依次运行 `prepare`、`lexical`、`semantic` 阶段，或运行 `all`。

数据集、模型、虚拟环境和实验输出不随源仓库提供，也不提交到本仓库。结果需要按小组版本重新生成，不能沿用旧实现的结果。
