import torch
from torch import nn
from torch.nn import functional as F


class Embedding_based(nn.Module):
    """TransE with normalized embeddings and distance-based BPR."""

    def __init__(self, args, n_entities, n_relations):
        super().__init__()
        if args.KG_embedding_type != "TransE":
            raise NotImplementedError("Only TransE is implemented.")
        if args.embed_dim != args.relation_dim:
            raise ValueError("TransE requires equal entity and relation dimensions.")
        self.entity_embed = nn.Embedding(n_entities, args.embed_dim)
        self.relation_embed = nn.Embedding(n_relations, args.relation_dim)
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    def query(self, h, r):
        return F.normalize(self.entity_embed(h)) + F.normalize(self.relation_embed(r))

    def score_TransE(self, h, r, t):
        return torch.linalg.vector_norm(self.query(h, r) - F.normalize(self.entity_embed(t)), dim=-1)

    calc_score = score_TransE

    def calc_kg_loss_TransE(self, h, r, pos_t, neg_t):
        vectors = [self.entity_embed(h), self.relation_embed(r),
                   self.entity_embed(pos_t), self.entity_embed(neg_t)]
        head, relation, pos, neg = map(F.normalize, vectors)
        query = head + relation
        pos_dist = torch.linalg.vector_norm(query - pos, dim=-1)
        neg_dist = torch.linalg.vector_norm(query - neg, dim=-1)
        penalty = sum(v.square().sum(dim=-1).mean() for v in vectors) / 2
        return F.softplus(pos_dist - neg_dist).mean() + self.kg_l2loss_lambda * penalty

    calc_loss = calc_kg_loss_TransE

    def forward(self, h, r, pos_t, neg_t, is_train=True):
        if is_train:
            return self.calc_loss(h, r, pos_t, neg_t)
        return self.calc_score(h, r, pos_t)
