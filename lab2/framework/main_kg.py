from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F

from KG_embedding_model import Embedding_based
from loader_kg import DataLoader
from metrics import kg_metrics_from_ranks
from parser_Embedding_based import parse_args
from log_helper import get_logger
from model_helper import load_model, save_checkpoint

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_kg(model, dataloader, device, Ks=(1, 3, 10), eval_batch_size=128,
                split="valid"):
    model.eval()
    frame = {"valid": dataloader.kg_valid_data, "test": dataloader.kg_test_data}[split]
    known = defaultdict(set)
    for facts in (dataloader.kg_train_data, dataloader.kg_valid_data, dataloader.kg_test_data):
        for h, r, t in facts.itertuples(index=False, name=None):
            known[h, r].add(t)

    entities = F.normalize(model.entity_embed.weight)
    ranks = []
    for start in range(0, len(frame), eval_batch_size):
        batch = frame.iloc[start:start + eval_batch_size]
        h, r, t = torch.as_tensor(batch.to_numpy(), device=device).T
        distances = torch.cdist(model.query(h, r), entities)
        for row, (head, relation, target) in enumerate(batch.itertuples(index=False, name=None)):
            others = list(known[head, relation] - {target})
            distances[row, others] = torch.inf
        target_distances = distances.gather(1, t[:, None])
        batch_ranks = 1 + (distances < target_distances).sum(dim=1)
        ranks.extend(batch_ranks.tolist())
    return kg_metrics_from_ranks(ranks, Ks)


def train_baseline(args, data):
    model = Embedding_based(args, data.n_entities, data.n_relations)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_mrr = -1
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        for _ in range(data.n_kg_data // args.kg_batch_size + 1):
            batch = data.generate_kg_batch(data.kg_dict, args.kg_batch_size, data.n_entities)
            loss = model(*batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % args.evaluate_every == 0 or epoch == args.n_epoch:
            current = evaluate_kg(model, data, "cpu", args.Ks, args.test_batch_size)["MRR"]
            logger.info("epoch=%d valid_MRR=%.4f", epoch, current)
            if current > best_mrr:
                best_mrr, best_state = current, deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model



def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    data = DataLoader(args)
    model = train_baseline(args, data)
    if args.checkpoint:
        save_checkpoint(model, args.checkpoint, embed_dim=args.embed_dim)
        load_model(model, args.checkpoint)
    scores = evaluate_kg(model, data, "cpu", args.Ks, args.test_batch_size, split="test")
    for name, value in scores.items():
        print(f"{name} = {value:.4f}")


if __name__ == "__main__":
    main()
