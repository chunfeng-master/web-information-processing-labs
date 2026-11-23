import pandas as pd
import torch
from torch.nn import functional as F
import numpy as np


# KG 链接预测指标

def get_ranks_from_scores(all_scores, true_indices, larger_is_better=False):

    if isinstance(all_scores, torch.Tensor):
        scores = all_scores.detach().cpu().numpy()
    else:
        scores = np.asarray(all_scores)

    true_indices = np.asarray(true_indices, dtype=np.int64)
    num_samples = scores.shape[0]

    ranks = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        s = scores[i]
        t_idx = true_indices[i]
        target_score = s[t_idx]

        if larger_is_better:
            better = np.sum(s > target_score)
        else:
            better = np.sum(s < target_score)

        ranks[i] = int(better) + 1  # rank 从 1 开始

    return ranks


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


def calc_kg_metrics(all_scores, true_indices, Ks=(1, 3, 10), larger_is_better=False):

    ranks = get_ranks_from_scores(all_scores, true_indices, larger_is_better=larger_is_better)
    metrics = kg_metrics_from_ranks(ranks, Ks=Ks)
    return metrics


class TrainEvidence:
    """Relation popularity and cosine neighbors derived exclusively from training facts."""

    def __init__(self, data):
        from scipy.sparse import csr_matrix

        train = data.kg_train_data.reset_index(drop=True)
        head, rel, tail = train.to_numpy().T
        self.n = data.n_entities
        self.counts = np.zeros((rel.max() + 1, self.n))
        np.add.at(self.counts, (rel, tail), 1)
        # A film is described by its observed (relation, tail) pairs.
        features = pd.factorize(pd.MultiIndex.from_arrays([rel, tail]))[0]
        matrix = csr_matrix((np.ones(len(head)), (head, features)), shape=(self.n, features.max()+1))
        df = np.asarray(matrix.sum(axis=0)).ravel()
        weighted = matrix.multiply(np.log1p(len(np.unique(head)) / df)).tocsr()
        norm = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
        weighted = weighted.multiply((1 / np.maximum(norm, 1e-12))[:, None]).tocsr()
        self.similarity = (weighted @ weighted.T).tocsr()
        self.similarity.setdiag(0)
        self.similarity.eliminate_zeros()
        self.tails = {}
        for r, group in train.groupby("r"):
            self.tails[r] = csr_matrix((np.ones(len(group)), (group.h, group.t)),
                                       shape=(self.n, self.n))

    def log_prior(self, h, r, neighbor_weight):
        count = self.counts[r]
        if neighbor_weight == 0:
            return np.log(count + 0.1)
        neighbors = self.similarity.getrow(h).power(2)
        votes = (neighbors @ self.tails[r]).toarray()[0]
        prior = (count + 0.1) / (count.sum() + 0.1 * self.n)
        return np.log(prior + neighbor_weight * votes)


class Context:
    """For each relation, predict frequent tails from other training relations of the film."""
    def __init__(self, data):
        from scipy.sparse import csr_matrix
        from sklearn.linear_model import LogisticRegression

        train = data.kg_train_data.reset_index(drop=True)
        features, vocab = pd.factorize(pd.MultiIndex.from_frame(train[["r", "t"]]))
        counts = np.bincount(features)
        self.design = csr_matrix((np.ones(len(train)), (train.h, features)),
                                 shape=(data.n_entities, len(vocab)))
        self.models = {}
        for rel, group in train.groupby("r"):
            keep = group.t.value_counts()
            group = group[group.t.isin(keep[keep >= 3].index)]
            columns = np.flatnonzero((vocab.get_level_values(0) != rel) & (counts >= 2))
            if group.t.nunique() < 2:
                continue
            design = self.design[:, columns]
            clf = LogisticRegression(C=2, max_iter=2000, solver="lbfgs")
            clf.fit(design[group.h.to_numpy()], group.t)
            prior = group.t.value_counts(normalize=True)[clf.classes_].to_numpy()
            self.models[rel] = (design, clf, prior)

    def adjustments(self, heads, rel, n):
        result = np.zeros((len(heads), n), dtype=np.float32)
        if rel in self.models:
            design, clf, prior = self.models[rel]
            result[:, clf.classes_] = clf.predict_log_proba(design[heads]) - np.log(prior)
        return result


class Cooccurrence:
    def __init__(self, data, ridge):
        from scipy.sparse import csr_matrix
        from scipy.linalg import cho_factor, cho_solve

        train = data.kg_train_data.reset_index(drop=True)
        pairs = pd.MultiIndex.from_frame(train[["r", "t"]])
        counts = pairs.value_counts()
        vocab = counts[counts >= 2].index
        indices = vocab.get_indexer(pairs)
        use = indices >= 0
        matrix = csr_matrix((np.ones(use.sum()), (train.h[use], indices[use])),
                            shape=(data.n_entities, len(vocab)))
        gram = (matrix.T @ matrix).toarray()
        gram.flat[::len(vocab)+1] += ridge
        precision = cho_solve(cho_factor(gram), np.eye(len(vocab)))
        coefficients = -precision / precision.diagonal()[None, :]
        np.fill_diagonal(coefficients, 0)
        self.values = matrix @ coefficients
        relations, tails = (vocab.get_level_values(k).to_numpy() for k in ("r", "t"))
        self.columns = {}
        for r in train.r.unique():
            cols = np.flatnonzero(relations == r)
            self.columns[r] = (tails[cols], cols)

    def scores(self, head, relation):
        tails, cols = self.columns[relation]
        return tails, self.values[head, cols]




class MaskedContext:
    """Predict a hidden tail from the remaining facts and relation cardinalities."""
    def __init__(self, data):
        from sklearn.ensemble import ExtraTreesClassifier

        from sklearn.linear_model import LogisticRegression

        train = data.kg_train_data.reset_index(drop=True)
        pairs = pd.MultiIndex.from_frame(train[["r", "t"]])
        counts = pairs.value_counts()
        vocab = counts[counts >= 2].index
        codes = vocab.get_indexer(pairs)
        use = codes >= 0
        h, r, t = train.to_numpy().T
        attributes = np.zeros((data.n_entities, len(vocab)), dtype=np.float32)
        attributes[h[use], codes[use]] = 1
        cardinalities = np.zeros((data.n_entities, r.max() + 1), dtype=np.float32)
        np.add.at(cardinalities, (h, r), 1)
        self.design = np.column_stack((attributes, cardinalities))
        self.models = {}
        for rel, group in train.groupby("r"):
            frequencies = group.t.value_counts()
            group = group[group.t.isin(frequencies[frequencies >= 3].index)]
            if group.t.nunique() < 2:
                continue
            observed = self.design[group.h].copy()
            observed[np.arange(len(group)), codes[group.index]] = 0
            observed[:, len(vocab) + rel] -= 1
            linear = LogisticRegression(C=0.6, max_iter=2000)
            forest = ExtraTreesClassifier(n_estimators=96, min_samples_leaf=3,
                                          max_features=0.6, random_state=2025)
            for clf in (linear, forest):
                clf.fit(observed, group.t)
            prior = frequencies[linear.classes_].to_numpy(dtype=float)
            prior /= prior.sum()
            self.models[rel] = linear, forest, prior

    def adjustments(self, heads, rel, n):
        result = np.zeros((len(heads), n), dtype=np.float32)
        if rel in self.models:
            linear, forest, prior = self.models[rel]
            observed = self.design[heads]
            prob = (linear.predict_proba(observed) + forest.predict_proba(observed)) / 2
            result[:, linear.classes_] = np.log(0.03 + 0.97 * prob / prior)
        return result


class RankingFeatures:
    def __init__(self, data):
        self.evidence = TrainEvidence(data)
        self.context = Context(data)
        self.cooccurrence = Cooccurrence(data, 3)
        self.masked_context = MaskedContext(data)

    @torch.no_grad()
    def transform(self, model, data, frame):
        """Build signals independently of the target, then prepare filtered ranks."""
        from collections import defaultdict

        n = data.n_entities
        known = defaultdict(set)
        for split in (data.kg_train_data, data.kg_valid_data, data.kg_test_data):
            for h, r, t in split.itertuples(index=False, name=None):
                known[h, r].add(t)
        signals = np.zeros((5, len(frame), n), dtype=np.float32)
        device = model.entity_embed.weight.device
        entities = F.normalize(model.entity_embed.weight)
        frame = frame.reset_index(drop=True)
        for rel, group in frame.groupby("r"):
            rows = group.index.to_numpy()
            heads = group.h.to_numpy()
            h = torch.as_tensor(heads, device=device)
            signals[0, rows] = -torch.cdist(model.query(h, torch.full_like(h, rel)), entities).cpu().numpy()
            signals[2, rows] = self.context.adjustments(heads, rel, n)
            signals[4, rows] = self.masked_context.adjustments(heads, rel, n)
            for row, head in zip(rows, heads):
                signals[1, row] = self.evidence.log_prior(head, rel, 0.3)
                tails, values = self.cooccurrence.scores(head, rel)
                signals[3, row, tails] = values
        targets = frame.t.to_numpy()
        # Exclude other known answers after scoring; the TransE coefficient is always one.
        for row, (h, r, t) in enumerate(frame.itertuples(index=False, name=None)):
            signals[0, row, list(known[h, r] - {t})] = -np.inf
        return signals, targets


def ranking_metrics(features, weights, ks):
    signals, targets = features
    score = signals[0].copy()
    for weight, signal in zip(weights, signals[1:]):
        if weight:
            score += weight * signal
    ranks = 1 + (score > score[np.arange(len(targets)), targets, None]).sum(axis=1)
    return kg_metrics_from_ranks(ranks, ks)


def format_metrics(metrics):
    return " | ".join(f"{name}={value:.5f}" for name, value in metrics.items())


def evaluate_fusion(model, data, ks):
    """Select fusion weights by validation MRR before computing test metrics."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        ranker = RankingFeatures(data)
        valid = ranker.transform(model, data, data.kg_valid_data)
        grids = [(0, .05, .15, .3, .6, 1.2), (0, .05, .15, .3, .6),
                 (0, .3, 1, 3, 6), (0, .05, .15, .3, .6, 1.2)]
        weights = [0.] * len(grids)
        stages = {"TransE": weights.copy()}
        for axis, name in enumerate(("neighbors", "context", "cooccurrence", "masked_context")):
            # Retune earlier coefficients as new evidence enters the score.
            for _ in range(2):
                for j in range(axis, -1, -1):
                    candidates = [weights[:j] + [w] + weights[j+1:] for w in grids[j]]
                    weights = max(candidates, key=lambda w: ranking_metrics(valid, w, ks)["MRR"])
            stages[name] = weights.copy()
            print("validation", name, weights, format_metrics(ranking_metrics(valid, weights, ks)))
        del valid
        test = ranker.transform(model, data, data.kg_test_data)
        for name in ("TransE", "masked_context"):
            print("test", name, stages[name], format_metrics(ranking_metrics(test, stages[name], ks)))
