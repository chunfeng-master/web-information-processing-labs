import logging
import os
import random
from ast import literal_eval
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn import functional as F

from KG_embedding_model import Embedding_based
from loader_kg import DataLoader
from log_helper import create_log_id, logging_config
from metrics import kg_metrics_from_ranks
from model_helper import early_stopping, load_model, save_model
from parser_Embedding_based import parse_args


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


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(2)

    # logging
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # load data
    data = DataLoader(args, logging)

    # construct model & optimizer
    model = Embedding_based(args, data.n_entities, data.n_relations)
    if args.use_pretrain == 1:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #  metrics & early stopping
    best_epoch = -1
    best_mrr = -1.0

    Ks = tuple(literal_eval(args.Ks))
    eval_batch_size = args.test_batch_size

    epoch_list = []
    # 记录每个 epoch 的 KG 指标
    kg_metrics_history = {
        'MR': [],
        'MRR': [],
    }
    for k in Ks:
        kg_metrics_history[f'Hits@{k}'] = []

    # train model
    edges = data.kg_data.to_numpy()
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        time1 = time()
        total_loss = 0.0

        order = np.random.permutation(len(edges)) if args.positive_sampling == "edge" else None
        n_batch = (len(edges) + data.kg_batch_size - 1) // data.kg_batch_size
        for iter in range(1, n_batch + 1):
            time2 = time()
            start = (iter - 1) * data.kg_batch_size
            size = min(data.kg_batch_size, len(edges) - start)
            if order is None:
                batch = data.generate_kg_batch(data.kg_dict, size, data.n_entities)
            else:
                h, r, t = edges[order[start:start + size]].T
                negatives = [data.sample_neg_triples_for_h(data.kg_dict, head, rel, 1, data.n_entities)[0]
                             for head, rel in zip(h, r)]
                batch = map(torch.as_tensor, (h, r, t, negatives))
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = batch
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            batch_loss = model(kg_batch_head,
                               kg_batch_relation,
                               kg_batch_pos_tail,
                               kg_batch_neg_tail,
                               is_train=True)

            if not torch.isfinite(batch_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {iter}")

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info(
                    'KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | '
                    'Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, iter, n_batch, time() - time2,
                        batch_loss.item(), total_loss / iter)
                )

        logging.info(
            'KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | '
            'Iter Mean Loss {:.4f}'.format(
                epoch, n_batch, time() - time1, total_loss / n_batch)
        )

        # ========= evaluate on validation KG =========
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time3 = time()
            metrics = evaluate_kg(model, data, device, Ks=Ks, eval_batch_size=eval_batch_size)

            log_str = 'KG Evaluation: Epoch {:04d} | Time {:.1f}s | MR {:.4f} | MRR {:.4f}'.format(
                epoch, time() - time3, metrics['MR'], metrics['MRR'])
            for k in Ks:
                log_str += ' | Hits@{} {:.4f}'.format(k, metrics[f'Hits@{k}'])
            logging.info(log_str)

            epoch_list.append(epoch)
            kg_metrics_history['MR'].append(metrics['MR'])
            kg_metrics_history['MRR'].append(metrics['MRR'])
            for k in Ks:
                kg_metrics_history[f'Hits@{k}'].append(metrics[f'Hits@{k}'])

            # 以 MRR 作为 early stopping 的指标
            cur_mrr_list = kg_metrics_history['MRR']
            best_mrr, should_stop = early_stopping(cur_mrr_list, args.stopping_steps)

            if should_stop:
                break

            # 如果当前 epoch 是目前为止 MRR 最好的，就保存模型
            if cur_mrr_list.index(best_mrr) == len(epoch_list) - 1:
                save_model(model, args.save_dir, log_save_id, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics to file
    metrics_df = {'epoch_idx': epoch_list}
    metrics_df['MR'] = kg_metrics_history['MR']
    metrics_df['MRR'] = kg_metrics_history['MRR']
    for k in Ks:
        metrics_df[f'Hits@{k}'] = kg_metrics_history[f'Hits@{k}']

    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv(os.path.join(args.save_dir, f'kg_metrics{log_save_id}.tsv'),
                      sep='\t', index=False)

    # print best metrics
    if best_epoch != -1:
        best_row = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
        log_str = (
            'Best KG Evaluation: Epoch {:04d} | MR {:.4f} | MRR {:.4f}'
            .format(int(best_row['epoch_idx']), best_row['MR'], best_row['MRR'])
        )
        for k in Ks:
            log_str += ' | Hits@{} {:.4f}'.format(k, best_row[f'Hits@{k}'])
        logging.info(log_str)
    else:
        logging.info('No best epoch found (maybe training stopped too early).')


def predict(args):

    # GPU / CPU
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # load data
    data = DataLoader(args, logging)

    # load model
    model = Embedding_based(args, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    torch.set_num_threads(2)

    Ks = (1, 3, 10) if not hasattr(args, "Ks") else tuple(literal_eval(args.Ks))
    eval_batch_size = args.test_batch_size

    if args.mode == "fusion":
        from metrics import evaluate_fusion
        evaluate_fusion(model, data, Ks)
        return
    metrics = evaluate_kg(model, data, device, Ks=Ks, eval_batch_size=eval_batch_size, split="test")
    print("Filtered test results:")
    print("MR  = {:.4f}".format(metrics['MR']))
    print("MRR = {:.4f}".format(metrics['MRR']))
    for k in Ks:
        print("Hits@{} = {:.4f}".format(k, metrics[f'Hits@{k}']))


if __name__ == '__main__':

    args = parse_args()
    (train if args.mode == 'train' else predict)(args)
