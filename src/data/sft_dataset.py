"""Conversation-level train/validation split and assistant-only VoyariLM SFT.

Preserves system, user, assistant and named tool messages. Assistant JSON is kept
verbatim. Prefixes are a plain-text convention; inference must use the same one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


DEFAULT_NAMES = (
    "01_voyari_v9_train.jsonl",
    "02_sgd_travel_clarification_train.jsonl",
)
FORMAT_VERSION = "voyari_sft_roles_v1"


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def role_header(message: dict) -> str:
    role = message["role"]
    if role == "tool" and message.get("name"):
        role += ":" + str(message["name"])
    return f"[{role}]\n"


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


@dataclass(frozen=True)
class ExampleRef:
    file_index: int
    byte_offset: int
    assistant_index: int
    answer_start: int


class SFTCorpus:
    """Build lightweight byte-offset indices; materialize one assistant turn on demand."""

    def __init__(
        self,
        paths: list[Path],
        tokenizer: Any,
        max_length: int = 1024,
        val_fraction: float = 0.02,
        seed: int = 42,
    ) -> None:
        if max_length < 256 or max_length > 1024:
            raise ValueError("max_length must be between 256 and 1024")
        if not 0 < val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")
        self.paths = [Path(path).resolve() for path in paths]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.val_fraction = val_fraction
        self.bos = tokenizer.token_to_id("<bos>")
        self.eos = tokenizer.token_to_id("<eos>")
        if self.bos is None or self.eos is None:
            raise ValueError("Tokenizer must contain <bos> and <eos>")

        # Leave at least ~25% of the window for user/system/tool context.
        self.target_chunk_size = max_length - max(128, max_length // 4)
        self.indices: dict[str, list[ExampleRef]] = {"train": [], "val": []}
        self.conversations: dict[str, int] = {"train": 0, "val": 0}
        self._streams: dict[int, Any] = {}
        self.file_hashes: dict[str, str] = {}
        self._index()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _split(self, file_index: int, line_number: int) -> str:
        key = f"{self.seed}:{self.paths[file_index].name}:{line_number}".encode()
        hash_value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return "val" if hash_value / 2**64 < self.val_fraction else "train"

    def _index(self) -> None:
        for fi, path in enumerate(self.paths):
            if not path.is_file():
                raise FileNotFoundError(f"SFT data file missing: {path}")
            self.file_hashes[path.name] = sha256_file(path)
            with path.open("rb") as stream:
                line_number = 0
                while True:
                    offset = stream.tell()
                    raw = stream.readline()
                    if not raw:
                        break
                    line_number += 1
                    if not raw.strip():
                        continue
                    try:
                        record = json.loads(raw)
                        messages = record["messages"]
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(f"Invalid JSONL: {path}:{line_number}") from exc
                    if not isinstance(messages, list):
                        raise ValueError(f"Expected messages list: {path}:{line_number}")
                    split = self._split(fi, line_number)
                    self.conversations[split] += 1
                    for assistant_index, message in enumerate(messages):
                        if message.get("role") != "assistant":
                            continue
                        content = text_content(message.get("content", ""))
                        if not content.strip():
                            continue
                        answer_length = len(self.encode(content)) + 1  # EOS
                        for start in range(0, answer_length, self.target_chunk_size):
                            self.indices[split].append(
                                ExampleRef(fi, offset, assistant_index, start)
                            )
        if not self.indices["train"] or not self.indices["val"]:
            raise RuntimeError("Train or validation split is empty")

    def _load(self, ref: ExampleRef) -> list[dict]:
        if ref.file_index not in self._streams:
            self._streams[ref.file_index] = self.paths[ref.file_index].open("rb")
        stream = self._streams[ref.file_index]
        stream.seek(ref.byte_offset)
        return json.loads(stream.readline())["messages"]

    def materialize(self, ref: ExampleRef) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self._load(ref)
        history = [self.bos]
        system_prefix = None
        for index, message in enumerate(messages[:ref.assistant_index]):
            role = message["role"]
            history.extend(self.encode(role_header(message)))
            history.extend(self.encode(text_content(message.get("content", ""))))
            if role == "assistant":
                history.append(self.eos)
            else:
                history.extend(self.encode("\n"))
            if index == 0 and role == "system":
                system_prefix = history.copy()

        response = text_content(messages[ref.assistant_index]["content"])
        answer = self.encode(response) + [self.eos]
        end = ref.answer_start + self.target_chunk_size
        target = answer[ref.answer_start:end]
        assert target, "Empty target segment"

        prompt = (
            history
            + self.encode("[assistant]\n")
            + answer[:ref.answer_start]
        )
        context_limit = self.max_length + 1 - len(target)
        if len(prompt) > context_limit:
            if system_prefix and len(system_prefix) < context_limit // 2:
                prompt = system_prefix + prompt[-(context_limit - len(system_prefix)):]
            else:
                prompt = [self.bos] + prompt[-(context_limit - 1):]
        if not prompt:
            raise RuntimeError("Assistant target missing context")

        token_ids = prompt + target
        # y[t] is the next token after x[t], with prompt positions ignored.
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor([-100] * (len(prompt) - 1) + target, dtype=torch.long)
        if len(x) != len(y) or x.numel() > self.max_length:
            raise RuntimeError("Bad SFT sample length or alignment")
        return x, y

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()


class SFTSplitDataset(Dataset):
    def __init__(self, corpus: SFTCorpus, split: str) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.corpus = corpus
        self.refs = corpus.indices[split]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.corpus.materialize(self.refs[index])
