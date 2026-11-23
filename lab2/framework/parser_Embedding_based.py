"""课程 TransE 实验的命令行参数，默认值对应报告中的 CPU 实验。"""
import argparse
import ast
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="TransE 尾实体预测")
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--data_name", default="freebase")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--relation_dim", type=int, default=32)
    parser.add_argument("--KG_embedding_type", choices=["TransE"], default="TransE")
    parser.add_argument("--kg_l2loss_lambda", type=float, default=1e-4)
    parser.add_argument("--kg_batch_size", type=int, default=2048)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epoch", type=int, default=40)
    parser.add_argument("--evaluate_every", type=int, default=5)
    parser.add_argument("--Ks", type=ast.literal_eval, default=[5, 10])
    parser.add_argument("--checkpoint", type=Path, help="保存最佳基础模型的位置")
    return parser.parse_args()
