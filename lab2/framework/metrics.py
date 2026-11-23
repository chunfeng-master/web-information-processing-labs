import numpy as np


def kg_metrics_from_ranks(ranks, Ks=(1, 3, 10)):

    ranks = np.asarray(ranks, dtype=np.float32)

    mr = ranks.mean()
    mrr = (1.0 / ranks).mean()

    metrics = {
        'MR': mr,
        'MRR': mrr,
    }
    for k in Ks:
        metrics[f'Hits@{k}'] = np.mean((ranks <= k).astype(np.float32))

    return metrics


