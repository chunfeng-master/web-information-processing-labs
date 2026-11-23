from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework"))

from main_kg import evaluate_kg, train_baseline
from parser_Embedding_based import parse_args
from model_helper import save_checkpoint
from KG_embedding_model import Embedding_based
from loader_kg import DataLoader
from ranking import Cooccurrence, evaluate, measures



def hard_negative_training(args, data):
    """Sample each directed training edge once per epoch with mixed negative tails."""
    rng = np.random.default_rng(args.seed)
    model = Embedding_based(args, data.n_entities, data.n_relations)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    edges = data.kg_data[["h", "r", "t"]].to_numpy()
    pools = {r: np.unique(group["t"]) for r, group in data.kg_data.groupby("r")}
    best = None
    for epoch in range(1, 61):
        model.train()
        order = rng.permutation(len(edges))
        for start in range(0, len(order), 512):
            h, r, t = edges[order[start:start + 512]].T
            neg = rng.integers(data.n_entities, size=len(h))
            hard = rng.random(len(h)) < 0.5
            for rel in np.unique(r[hard]):
                use = hard & (r == rel)
                neg[use] = rng.choice(pools[rel], size=use.sum())
            for i, (head, rel) in enumerate(zip(h, r)):
                while neg[i] in data.positive_tails[head, rel]:
                    neg[i] = rng.integers(data.n_entities)
            loss = model(*(torch.from_numpy(x) for x in (h, r, t, neg)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            valid = evaluate_kg(model, data, "cpu", (5, 10), 64)
            if best is None or valid["MRR"] > best[0]:
                best = (valid["MRR"], epoch, deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    return model, best[1]


def select_weights(model, data, known, weights, addon=None):
    rows = evaluate(model, data, data.kg_valid_data, weights, known, addon)
    metrics = {key: measures(predictions) for key, predictions in rows.items()}
    key = max(metrics, key=lambda item: metrics[item]["MRR"])
    return key, metrics[key]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    data = DataLoader(args)
    splits = (data.kg_train_data, data.kg_valid_data, data.kg_test_data)
    if len(pd.concat(splits).drop_duplicates()) != sum(map(len, splits)):
        raise ValueError("数据划分存在重复三元组")
    known = defaultdict(set)
    for frame in splits:
        for h, r, t in frame.itertuples(index=False, name=None):
            known[h, r].add(t)

    baseline = train_baseline(args, data)
    mixed, _ = hard_negative_training(args, data)
    direct = (0, 0, 0)
    weights = [direct] + [(s, n, 0) for s in (0.05, 0.15, 0.5, 1.5)
                          for n in (0, 0.03, 0.3, 3)]
    models = [baseline, mixed]
    choices = [select_weights(model, data, known, weights) for model in models]
    index = max(range(len(models)), key=lambda i: choices[i][1]["MRR"])
    model = models[index]
    prior, valid_b = choices[index]
    context_grid = [prior] + [(s, n, c) for s in (0.075, 0.15, 0.3)
                             for n in (0.03, 0.3, 3) for c in (0.05, 0.15, 0.3)]
    context, valid_c = select_weights(model, data, known, context_grid)
    ease_grid = [context + (w,) for w in (0, 0.3, 1.0, 3.0)]
    candidates = []
    for ridge in (3, 30, 100):
        weights_d, valid_d = select_weights(model, data, known, ease_grid, Cooccurrence(data, ridge))
        candidates.append((ridge, weights_d, valid_d))
    ridge, selected, valid_d = max(candidates, key=lambda row: row[2]["MRR"])

    _, valid_a = select_weights(baseline, data, known, [direct])
    methods = [
        ("A_TransE", baseline, direct, None, valid_a),
        ("B_neighbors", model, prior, None, valid_b),
        ("C_context", model, context, None, valid_c),
        ("D_cooccurrence", model, selected, Cooccurrence(data, ridge), valid_d),
    ]
    metrics, predictions = [], []
    for name, fitted, key, addon, val in methods:
        rows = evaluate(fitted, data, data.kg_test_data, [key], known, addon)[key]
        frame = pd.DataFrame(rows, columns=["h", "r", "t", "top1", "rank"])
        pd.testing.assert_frame_equal(frame[["h","r","t"]], data.kg_test_data)
        predictions.append(frame.assign(method=name))
        metrics.append(dict(method=name, valid_MRR=val["MRR"], **measures(rows)))

    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    pd.concat(predictions).to_csv(output/"predictions.tsv", sep="\t", index=False)
    pd.DataFrame(metrics).to_csv(output/"metrics.tsv", sep="\t", index=False)
    save_checkpoint(model, output/"final_model.pth", weights=selected, ridge=ridge, embed_dim=args.embed_dim)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
