from __future__ import annotations

import argparse
import os
import sys
import threading
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
        has_ple: bool = False,
        final_norm: Optional[np.ndarray] = None,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.lm_head = lm_head
        self.embedding = embedding
        self.seq_len = seq_len
        self.has_ple = has_ple
        self.final_norm = final_norm

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        return self.embedding[input_ids]

    def _rms_norm(self, x, eps=1e-6):
        variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        x_norm = x / np.sqrt(variance + eps)
        return (x_norm * self.final_norm).astype(np.float32)

    def _compute_logits(self, hidden_states: np.ndarray) -> np.ndarray:
        if self.final_norm is not None:
            hidden_states = self._rms_norm(hidden_states)
        return hidden_states @ self.lm_head.T

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float = 0.7) -> np.ndarray:
        if temperature < 0.01:
            return np.argmax(logits, axis=-1)
        logits = logits.astype(np.float64) / temperature
        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return np.array([[np.random.choice(probs.shape[-1], p=probs[0])]])

    def generate(self, prompt: str, max_tokens: int = 1, stream: bool = False, temperature: float = 0.7) -> str:
        # Apply chat template for instruct models
        if hasattr(self.tokenizer, "apply_chat_template") and not prompt.startswith("<|"):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt

        # Tokenize without padding first to get true length
        raw = self.tokenizer(formatted, return_tensors="np", add_special_tokens=False)
        true_len = raw["input_ids"].shape[1]

        # Left-pad to self.seq_len (causal LMs need left-padding so the last token is real)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        if true_len >= self.seq_len:
            input_ids = raw["input_ids"][:, -self.seq_len:].astype(np.int32)
        else:
            pad_len = self.seq_len - true_len
            padded = np.full((1, self.seq_len), pad_id, dtype=np.int32)
            padded[0, pad_len:] = raw["input_ids"][0, :true_len]
            input_ids = padded.astype(np.int32)

        output_pieces = []
        kv_cache = {}

        # ---- Prefill: full 64-token forward pass ----
        x = self._embed(input_ids)
        if self.has_ple:
            hidden_states = self.root.run(x, kv_cache=kv_cache, prefill=True,
                                           input_ids=input_ids)
        else:
            hidden_states = self.root.run(x, kv_cache=kv_cache, prefill=True)
        logits = self._compute_logits(hidden_states)
        next_token = self._sample(logits[:, -1, :], temperature)

        piece = self.tokenizer.decode([int(next_token[0, 0])])
        output_pieces.append(piece)
        if stream:
            print(piece, end="", flush=True)

        eos_id = self.tokenizer.eos_token_id

        generated = input_ids.copy()

        # ---- Decode steps: single-token KV-cached forward passes ----
        for step in range(max_tokens - 1):
            generated = np.concatenate(
                [generated, next_token.reshape(1, 1)], axis=1
            )

            # Embed only the new token
            x = self._embed(next_token.reshape(1, 1))

            if self.has_ple:
                hidden_states = self.root.run(
                    x, kv_cache=kv_cache, prefill=False, input_ids=generated,
                )
            else:
                hidden_states = self.root.run(
                    x, kv_cache=kv_cache, prefill=False,
                )
            logits = self._compute_logits(hidden_states)
            # For single-token decode, the only position is position 0
            next_token = self._sample(logits[:, 0, :], temperature)

            if eos_id is not None and int(next_token[0, 0]) == eos_id:
                break

            piece = self.tokenizer.decode([int(next_token[0, 0])])
            output_pieces.append(piece)
            if stream:
                print(piece, end="", flush=True)

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


def _build_gen(
    worker_addrs: List[Tuple[str, int]],
    model: str = "HuggingFaceTB/SmolLM-135M",
    num_layers: int = 0,
    real_weights: bool = False,
) -> Generator:
    from transformers import AutoTokenizer

    from src.orchestrator.llama_loader import WeightProvider

    config = ModelConfig.from_hf(model)

    if num_layers <= 0:
        num_layers = config.num_layers

    total_nodes = 1 + len(worker_addrs)
    ffn_chunk = config.ffn_dim // total_nodes
    seq_chunk = 64 // total_nodes
    partitions = {}
    for i in range(total_nodes):
        ffn_start = i * ffn_chunk
        ffn_end = (i + 1) * ffn_chunk if i < total_nodes - 1 else config.ffn_dim
        seq_start = i * seq_chunk
        seq_end = (i + 1) * seq_chunk if i < total_nodes - 1 else 64
        partitions[i] = {
            "ffn_start": ffn_start,
            "ffn_end": ffn_end,
            "seq_start": seq_start,
            "seq_end": seq_end,
        }

    wp = WeightProvider(model, partitions, num_layers=num_layers)

    all_layer_weights = {}
    for node_id in range(total_nodes):
        if node_id == 0:
            all_layer_weights[node_id] = wp.get_root_weights(node_id)
        else:
            all_layer_weights[node_id] = wp.get_node_weights(node_id, total_nodes)

    has_ple = wp.ple_dim > 0
    root = RootNode(
        worker_addrs, config, all_layer_weights,
        ple_embedding=wp.get_ple_embedding() if has_ple else None,
        ple_projection=wp.get_ple_projection() if has_ple else None,
        ple_projection_norm=wp.get_ple_projection_norm() if has_ple else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return Generator(
        root, tokenizer,
        lm_head=wp.get_lm_head(),
        embedding=wp.get_embedding(),
        seq_len=64,
        has_ple=has_ple,
        final_norm=wp.get_final_norm(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Llama Generator")
    parser.add_argument(
        "--workers", type=str, default=None,
        help="Comma-separated list of worker IP:PORT. If omitted, uses mDNS auto-discovery.",
    )
    parser.add_argument(
        "--model", type=str, default="HuggingFaceTB/SmolLM-135M",
        help="HuggingFace model ID (e.g. mistralai/Mistral-7B-v0.3, google/gemma-4-2b-it)",
    )
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=10, help="Number of tokens to generate")
    parser.add_argument("--layers", type=int, default=0, help="Number of transformer layers (0 = auto, all layers)")
    parser.add_argument(
        "--local-worker", action="store_true",
        help="Start a worker on this machine (avoids needing a separate terminal)",
    )
    parser.add_argument(
        "--discover-timeout", type=float, default=3.0,
        help="Seconds to wait for mDNS worker discovery",
    )
    parser.add_argument(
        "--expect-workers", type=int, default=None,
        help="Expected number of workers (mDNS returns as soon as this many are found)",
    )
    args = parser.parse_args()

    if args.workers:
        worker_addrs = _parse_workers(args.workers)
    else:
        from src.orchestrator.mdns import discover_workers
        worker_addrs = discover_workers(
            timeout=args.discover_timeout,
            expect=args.expect_workers,
        )
        if not worker_addrs:
            # If no remote workers and --local-worker, that's fine — we'll add one
            if not args.local_worker:
                print("ERROR: No workers discovered. Start workers first or use --workers.")
                sys.exit(1)

    local_worker = None
    if args.local_worker:
        from src.orchestrator.worker_node import WorkerNode
        local_worker = WorkerNode(host="localhost", port=0, use_mdns=False)
        ready = threading.Event()
        t = threading.Thread(target=local_worker.start, kwargs={"ready_event": ready}, daemon=True)
        t.start()
        ready.wait()
        worker_addrs.insert(0, ("localhost", local_worker.port))

    gen = _build_gen(worker_addrs, model=args.model, num_layers=args.layers)

    try:
        gen.generate(args.prompt, max_tokens=args.max_tokens, stream=True)
    finally:
        gen.root.shutdown()
