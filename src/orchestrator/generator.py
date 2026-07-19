from __future__ import annotations

from typing import Optional

import numpy as np

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

    def generate(self, prompt: str, max_tokens: int = 1) -> str:
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

        for _ in range(max_tokens):
            x = self._embed(generated)
            hidden_states = self.root.run(x)
            logits = self._compute_logits(hidden_states)
            next_token_logits = logits[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1)

            generated = np.concatenate(
                [generated, next_token.reshape(1, 1)], axis=1
            )
            generated = generated[:, -self.seq_len:]

        output_ids = generated[0].tolist()
        if pad_id is not None:
            decoded = self.tokenizer.decode(
                [t for t in output_ids if t != pad_id]
            )
        else:
            decoded = self.tokenizer.decode(output_ids)

        return decoded
