import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import ESMM_DIN_DCN
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(schema, per_position_vocab_sizes):
    specs = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args():
    parser = argparse.ArgumentParser(description="ESMM+DIN+DCN Training")

    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--schema_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--buffer_batches', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512')

    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--num_cross_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.1)

    parser.add_argument('--loss_type', type=str, default='focal', choices=['bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--sparse_lr', type=float, default=0.05)
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0)
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1)
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0)

    parser.add_argument('--emb_skip_threshold', type=int, default=0)
    parser.add_argument('--seq_id_threshold', type=int, default=10000)

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups)
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer')
    parser.add_argument('--user_ns_tokens', type=int, default=0)
    parser.add_argument('--item_ns_tokens', type=int, default=0)

    args = parser.parse_args()

    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main():
    args = parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    schema_path = args.schema_path or os.path.join(args.data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())

    logging.info("Loading data...")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
    else:
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_cross_layers": args.num_cross_layers,
        "dropout_rate": args.dropout_rate,
    }

    model = ESMM_DIN_DCN(**model_args).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"ESMM_DIN_DCN created: {total_params:,} params")

    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {"layer": 1, "head": 1, "hidden": args.d_model}

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        train_config=vars(args),
    )

    trainer.train()
    writer.close()
    logging.info("Training complete!")


if __name__ == '__main__':
    main()
