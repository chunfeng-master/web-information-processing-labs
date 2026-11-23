import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

from metrics import kg_metrics_from_ranks


def measures(predictions):
    return {k: float(v) for k, v in kg_metrics_from_ranks(
        [row[-1] for row in predictions], (5, 10)).items()}


class TrainEvidence:
    """Relation popularity and cosine neighbors derived exclusively from training facts."""

    def __init__(self, data):
        train = data.kg_train_data
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
        train = data.kg_train_data
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
            clf = LogisticRegression(C=2, max_iter=250, solver="lbfgs")
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
        train = data.kg_train_data
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


def evaluate(model, data, frame, candidates, known, cooccurrence=None):
    evidence, context = TrainEvidence(data), Context(data)
    rows = {key: [] for key in candidates}
    entities = F.normalize(model.entity_embed.weight).detach()
    with torch.no_grad():
        for r, group in frame.groupby("r"):
            heads = group.h.to_numpy()
            h = torch.from_numpy(heads)
            distances = torch.cdist(model.query(h, torch.full_like(h, r)), entities).numpy()
            adjustments = context.adjustments(heads, r, data.n_entities)
            for i, (idx, head, target) in enumerate(zip(group.index, heads, group.t)):
                blocked = list(known[head, r] - {target})
                priors = {nw:evidence.log_prior(head, r, nw) for nw in {k[1] for k in candidates}}
                for key in candidates:
                    scale, nw, extra = key[:3]
                    score = -distances[i] + scale * priors[nw] + extra * adjustments[i]
                    if cooccurrence is not None and key[3]:
                        tails, values = cooccurrence.scores(head, r)
                        score[tails] += key[3] * values
                    score[blocked] = -np.inf
                    rank = int(1 + (score > score[target]).sum())
                    rows[key].append((idx, head, r, target, int(score.argmax()), rank))
    return {key:[row[1:] for row in sorted(values)] for key, values in rows.items()}
