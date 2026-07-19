from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from src.orchestrator.cluster import ModelConfig
from src.orchestrator.root_node import RootNode


class Generator:
    def __init__(
        self,
        root: RootNode,
        tokenizer,
        lm_head: np.ndarray,
        embedding: np.ndarray,
        seq_len: int = 64,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.lm_head = lm_head
        self.embedding = embedding
        self.seq_len = seq_len

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        return self.embedding[input_ids]

    def _compute_logits(self, hidden_states: np.ndarray) -> np.ndarray:
        return hidden_states @ self.lm_head.T

    def generate(self, prompt: str, max_tokens: int = 1, stream: bool = False) -> str:
        tokens = self.tokenizer(
            prompt, return_tensors="np",
            padding="max_length", max_length=self.seq_len,
            truncation=True,
        )
        input_ids = tokens["input_ids"].astype(np.int32)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        generated = input_ids.copy()
        output_pieces = []

        for step in range(max_tokens):
            x = self._embed(generated)
            hidden_states = self.root.run(x)
            logits = self._compute_logits(hidden_states)
            next_token_logits = logits[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1)

            piece = self.tokenizer.decode([int(next_token[0])])
            output_pieces.append(piece)
            if stream:
                print(piece, end="", flush=True)

            generated = np.concatenate(
                [generated, next_token.reshape(1, 1)], axis=1
            )
            generated = generated[:, -self.seq_len:]

        if stream:
            print()

        return "".join(output_pieces)


def _parse_workers(workers_str: str) -> List[Tuple[str, int]]:
    pairs = []
    for part in workers_str.split(","):
        part = part.strip()
        if not part:
            continue
        host, port_str = part.rsplit(":", 1)
        pairs.append((host, int(port_str)))
    return pairs


def _build_gen(worker_addrs: List[Tuple[str, int]]) -> Generator:
    from transformers import AutoTokenizer

    from src.orchestrator.llama_loader import create_synthetic_weights, get_smollm_config

    smollm_config = get_smollm_config()
    model_config = ModelConfig(
        hidden_dim=smollm_config["hidden_dim"],
        n_heads=smollm_config["n_heads"],
        n_kv_heads=smollm_config["n_kv_heads"],
        head_dim=smollm_config["head_dim"],
        ffn_dim=smollm_config["ffn_dim"],
    )

    root = RootNode(worker_addrs, model_config)
    full_weights = create_synthetic_weights(smollm_config)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return Generator(
        root, tokenizer,
        lm_head=full_weights["lm_head"],
        embedding=full_weights["embedding"],
        seq_len=64,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Llama Generator")
    parser.add_argument(
        "--workers", type=str, required=True,
        help="Comma-separated list of worker IP:PORT (e.g. 192.168.1.50:9000,192.168.1.51:9000)",
    )
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=10, help="Number of tokens to generate")
    args = parser.parse_args()

    worker_addrs = _parse_workers(args.workers)
    gen = _build_gen(worker_addrs)
    gen.generate(args.prompt, max_tokens=args.max_tokens, stream=True)
