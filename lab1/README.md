# Web 信息处理与应用：实验 1

本目录包含实验 1 必做部分的完整实现。`Qilin-IR-50K/optional/` 未使用。

## 目录

- `src/lab1_ir.py`：分词、规范化、倒排索引、跳表、压缩和检索实现
- `run_experiment.py`：运行全部实验并生成可复现结果
- `tests/test_lab1_ir.py`：核心算法单元测试
- `outputs/`：运行后生成索引、指标、案例和报告素材
- `实验报告.md`：正式 Markdown 实验报告，内容严格按照实验文档中的必做项顺序编排（提交前需填写成员信息与科大云盘链接）

`outputs/README.md` 解释每个结果文件。`outputs/results.json` 的第一个字段 `_README` 解释各级字段、单位和实验局限，助教无需先阅读代码即可理解结果。

## 运行

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
python verify_completion.py
```

第一次运行语义检索时会下载 `paraphrase-multilingual-MiniLM-L12-v2`。若只想先完成非语义部分，可运行：

```powershell
python run_experiment.py --skip-semantic
```

可用 `--max-docs 5000` 做快速试跑；正式结果默认使用全部 50,000 篇文档。固定随机种子为 2026。

正式运行会生成：

- HMM 与双向最大匹配的全量分词结果，各自包含保留和去除停用词两套设置；
- 未压缩位置倒排索引和可查询的 gap+VB 压缩索引；
- 布尔查询顺序、跳表步长、短语检索、压缩率和端到端查询计时；
- TF-IDF、完整段落编码、逐词编码后平均以及停用词对照结果。

