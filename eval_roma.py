#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import BertTokenizer

from lib.datasets import image_caption_bert, image_caption_bigru
from lib.evaluation import compute_sims, encode_data
from lib.model import Model
from lib.roma_config import require_bert_path, vocab_filename
from lib.roma_evaluation import DEFAULT_KS, text_to_scene_metrics
from lib.roma_runs import format_markdown_metrics
from lib.vocab import deserialize_vocab
from train import _encode_scene_features


def build_loader(opt):
    if opt.text_enc_type == "bigru":
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_filename(opt.data_name)))
        vocab.add_word("<mask>")
        opt.vocab_size = len(vocab)
        opt.word2idx = vocab.word2idx if opt.wemb_type is not None else None
        return image_caption_bigru.get_test_loader(
            "val", opt.data_name, vocab, opt.batch_size, opt.workers, opt
        )
    opt.bert_path = str(require_bert_path(opt.bert_path))
    tokenizer = BertTokenizer.from_pretrained(opt.bert_path, local_files_only=True)
    opt.vocab_size = len(tokenizer.vocab)
    opt.word2idx = None
    return image_caption_bert.get_test_loader(
        "val", opt.data_name, tokenizer, opt.batch_size, opt.workers, opt
    )


def evaluate(model_path: str, data_root: str, output_json: str | None = None) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    opt = checkpoint["opt"]
    opt.data_path = data_root
    opt.data_root = data_root
    opt.workers = int(getattr(opt, "workers", 10))
    loader = build_loader(opt)
    model = Model(opt)
    model.load_state_dict(checkpoint["model"])
    model.val_start()
    with torch.no_grad():
        _, cap_embs, _, cap_lens, caption_scene_indices = encode_data(
            model, loader, return_scene_indices=True
        )
        img_embs, img_lens = _encode_scene_features(model, loader.dataset.images, opt.batch_size)
        # A 128x128 cross-attention shard can exceed 24 GB for 200-region RoMa
        # scenes. Smaller shards compute identical scores with bounded memory.
        sims = compute_sims(img_embs, cap_embs, img_lens, cap_lens, model, shard_size=32)
    metrics = text_to_scene_metrics(sims.T, caption_scene_indices, ks=DEFAULT_KS)
    result = {
        "dataset": opt.data_name,
        "text_encoder": opt.text_enc_type,
        "checkpoint": str(Path(model_path).resolve()),
        "queries": len(caption_scene_indices),
        "scenes": len(loader.dataset.feature_scene_ids),
        "metrics": metrics,
    }
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        output_path.with_suffix(".md").write_text(
            format_markdown_metrics(opt.data_name, opt.text_enc_type, metrics), encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CHAN on a RoMa text-to-3D dataset.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_path, args.data_root, args.output_json), indent=2))


if __name__ == "__main__":
    main()
