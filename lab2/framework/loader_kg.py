import gzip
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path

import os
import random
import collections

import torch
import numpy as np
import pandas as pd
from copy import deepcopy


class DataLoaderBase(object):

    def __init__(self, args, logging=None):
        self.args = args
        self.data_name = args.data_name

        self.data_dir = os.path.join(args.data_dir, args.data_name)

        # 直接使用已经划分好的 3 个文件
        self.kg_train_file = os.path.join(self.data_dir, "train.txt")
        self.kg_valid_file = os.path.join(self.data_dir, "valid.txt")
        self.kg_test_file  = os.path.join(self.data_dir, "test.txt")

        # 读取三份 KG（DataFrame）
        self.kg_train_data = self.load_kg(self.kg_train_file)
        self.kg_valid_data = self.load_kg(self.kg_valid_file)
        self.kg_test_data  = self.load_kg(self.kg_test_file)

        if logging:
            logging.info(f"KG train triples: {len(self.kg_train_data)}")
            logging.info(f"KG valid triples: {len(self.kg_valid_data)}")
            logging.info(f"KG test  triples: {len(self.kg_test_data)}")
        else:
            print(f"KG train triples: {len(self.kg_train_data)}")
            print(f"KG valid triples: {len(self.kg_valid_data)}")
            print(f"KG test  triples: {len(self.kg_test_data)}")


    def load_kg(self, filename):
        """
        读取一个 KG 文件为 DataFrame，列名为 ['h', 'r', 't']。
        格式假定：txt文件每行 "h r t"，用空格分隔。
        """
        kg_data = pd.read_csv(
            filename,
            sep=' ',
            names=['h', 'r', 't'],
            engine='python'
        )
        kg_data = kg_data.drop_duplicates()
        return kg_data


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        """
        为head采样正样本。

        """
        pairs = random.sample(kg_dict[head], n_sample_pos_triples)
        return [r for t, r in pairs], [t for t, r in pairs]

    def sample_neg_triples_for_h(self, kg_dict, head, relation,
                                 n_sample_neg_triples, highest_neg_idx):
        if kg_dict is getattr(self, "kg_dict", None):
            positives = self.positive_tails.get((head, relation), set())
        else:
            positives = {t for t, r in kg_dict[head] if r == relation}
        if n_sample_neg_triples > highest_neg_idx - len(positives):
            raise ValueError("Insufficient distinct negative tails")
        pool = self.relation_tails.get(relation, ()) if kg_dict is getattr(self, 'kg_dict', None) else ()
        sampled = []
        while len(sampled) < n_sample_neg_triples:
            tail = random.randrange(highest_neg_idx)
            if pool and random.random() < self.args.relation_negative:
                candidate = random.choice(pool)
                if candidate < highest_neg_idx and candidate not in positives and candidate not in sampled:
                    tail = candidate
            if tail not in positives and tail not in sampled:
                sampled.append(tail)
        return sampled

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """
        根据当前 kg_dict（由训练集 + 反向边构成）生成一个 KG batch：

        """
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(list(exist_heads), batch_size)
        else:
            batch_head = np.random.choice(list(exist_heads),
                                          batch_size,
                                          replace=True).tolist()

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(
                kg_dict, h, relation[0], 1, highest_neg_idx
            )
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


class DataLoader(DataLoaderBase):
    """
    在 DataLoaderBase 的基础上：
      - 使用 train KG 构建 self.kg_data（带反向边，用于训练 & 采样。即这是真实训练时用的训练集。）
      - 用 train+valid+test 统计 n_entities / n_relations
      - 构建 kg_dict / relation_dict
    """

    def __init__(self, args, logging=None):
        super().__init__(args, logging)

        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        self.construct_data()
        self.print_info(logging)

    def construct_data(self):
        """
        在train KG上添加反向三元组，构建训练图self.kg_data；并统计所有数据中的实体数、关系数。

        注：因为由（h,r,t）添加了反向三元组(t,r',h)，且我们此处认为，r和r'是不同的关系，
        所以建模时关系数会因为添加反向三元组而翻倍。
        """

        all_kg_list = [self.kg_train_data, self.kg_valid_data, self.kg_test_data]
        all_kg = pd.concat(all_kg_list, axis=0, ignore_index=True)

        max_ent_id = int(max(all_kg['h'].max(), all_kg['t'].max()))
        self.n_entities = max_ent_id + 1

        max_rel_id = int(all_kg['r'].max())
        n_relations_origin = max_rel_id + 1

        kg_train = self.kg_train_data
        self.n_kg_train = len(kg_train)

        reverse_kg_data = deepcopy(kg_train)
        reverse_kg_data['h'], reverse_kg_data['t'] = reverse_kg_data['t'], reverse_kg_data['h']
        reverse_kg_data['r'] = reverse_kg_data['r'] + n_relations_origin

        self.kg_data = pd.concat([kg_train, reverse_kg_data],
                                 axis=0, ignore_index=True)

        max_rel_id_all = int(self.kg_data['r'].max())
        self.n_relations = max_rel_id_all + 1

        self.n_kg_data = len(self.kg_data)
        self.n_kg_valid = len(self.kg_valid_data)
        self.n_kg_test = len(self.kg_test_data)

        self.kg_dict = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)

        self.positive_tails = collections.defaultdict(set)
        for h, r, t in self.kg_data.itertuples(index=False, name=None):
            self.kg_dict[h].append((t, r))
            self.positive_tails[h, r].add(t)
            self.relation_dict[r].append((h, t))


        self.relation_tails = {r: sorted({t for _, t in pairs})
                               for r, pairs in self.relation_dict.items()}

    def print_info(self, logging=None):
        if logging:
            logging.info('n_entities:   %d' % self.n_entities)
            logging.info('n_relations:  %d' % self.n_relations)
            logging.info('n_kg_train:   %d' % self.n_kg_train)
            logging.info('n_kg_valid:   %d' % self.n_kg_valid)
            logging.info('n_kg_test:    %d' % self.n_kg_test)
            logging.info('n_kg_data(含反向): %d' % self.n_kg_data)
        else:
            print('n_entities:   %d' % self.n_entities)
            print('n_relations:  %d' % self.n_relations)
            print('n_kg_train:   %d' % self.n_kg_train)
            print('n_kg_valid:   %d' % self.n_kg_valid)
            print('n_kg_test:    %d' % self.n_kg_test)
            print('n_kg_data(含反向): %d' % self.n_kg_data)


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
MOVIES = 1600
NS = rb"<http://rdf\.freebase\.com/ns/"
MID = rb"([mg]\.[A-Za-z0-9_]+)>"
FILM_EDGE = re.compile(b"^" + NS + MID + rb"\t" + NS + rb"(film\.film\.[^>]+)>\t" + NS + MID + rb"\t\.[\r]?$", re.M)


def sample_movies(path):
    # 保留哈希值最小的电影；阈值只会下降，最终入选电影的早期边也被完整保留。
    selected, heap = {}, []
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(8 << 20):
            chunk += stream.readline()
            for match in FILM_EDGE.finditer(chunk):
                head, relation, tail = match.groups()
                if head not in selected:
                    digest = hashlib.blake2b(head, key=str(SEED).encode(), digest_size=16).digest()
                    priority = int.from_bytes(digest, "big")
                    if len(heap) == MOVIES:
                        if priority >= -heap[0][0]:
                            continue
                        del selected[heapq.heappop(heap)[1]]
                    heapq.heappush(heap, (-priority, head))
                    selected[head] = set()
                selected[head].add((relation, tail))
    return sorted((h.decode(), r.decode(), t.decode())
                  for h, edges in selected.items() for r, t in edges)


def prepare_data():
    triples = sample_movies(ROOT / "data/raw/freebase_douban.gz")
    entities = sorted({v for h, _, t in triples for v in (h, t)})
    relations = sorted({r for _, r, _ in triples})
    if len(entities) < 3000:
        raise ValueError("电影子图实体不足 3000")
    output = ROOT / "data/freebase"
    output.mkdir(parents=True, exist_ok=True)
    eid = {v: i for i, v in enumerate(entities)}
    rid = {v: i for i, v in enumerate(relations)}
    rows = [(eid[h], rid[r], eid[t]) for h, r, t in triples]
    random.Random(SEED).shuffle(rows)
    # 仅在训练集仍覆盖每个实体和关系时留出该边。
    counts = Counter(v for h, _, t in rows for v in {h, t})
    rel_counts = Counter(r for _, r, _ in rows)
    train, held = [], []
    per_split = len(rows) // 10
    for h, r, t in rows:
        endpoints = {h, t}
        if len(held) < 2 * per_split and min(counts[v] for v in endpoints) > 1 and rel_counts[r] > 1:
            held.append((h, r, t))
            counts.subtract(endpoints)
            rel_counts[r] -= 1
        else:
            train.append((h, r, t))
    if len(held) != 2 * per_split:
        raise ValueError("无法在保持训练集覆盖率时完成划分")
    random.Random(SEED + 1).shuffle(held)
    splits = dict(train=train, valid=held[:per_split], test=held[per_split:])
    for name, split in splits.items():
        (output / f"{name}.txt").write_text(
            "".join(f"{h} {r} {t}\n" for h, r, t in split), encoding="utf-8")
    stats = dict(seed=SEED, films=len({h for h, _, _ in triples}), entities=len(eid),
                 relations=len(rid), triples=len(rows), splits={k: len(v) for k, v in splits.items()})
    print(json.dumps(stats))



if __name__ == "__main__":
    prepare_data()
