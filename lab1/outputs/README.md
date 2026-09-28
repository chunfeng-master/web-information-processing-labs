# 实验结果文件说明

本目录只包含实验生成物，不包含原始数据。助教建议先看 `results.json` 顶部的 `_README`，再结合实验报告阅读。

- `results.json`：全部实验的机器可读结果、查询案例、耗时、压缩率和检索结果。
- `inverted_index_full.sqlite`：未压缩位置倒排索引，含 `docs`、`postings`、`meta` 三张表。
- `compressed_index_full.sqlite`：持久化压缩索引。docID 使用 gap+VB，tf、位置计数和位置间距使用 VB。
- `normalized_tokens_full.jsonl`：HMM 分词的保留/去除停用词两套结果，一行一篇文档。
- `matching_tokens_full.jsonl`：双向最大匹配分词的保留/去除停用词两套结果。

计时单位为毫秒，空间单位为字节。点击指标只作为相关性代理，不能把未点击文档直接视为不相关。
