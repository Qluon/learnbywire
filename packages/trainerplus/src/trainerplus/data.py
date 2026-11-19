"""CPT/SFT preparation using public tokenizer APIs; no model-specific template."""
import csv
import hashlib
import json
from pathlib import Path
import re

import torch
from .config import DataConfig


class TokenDataset(torch.utils.data.Dataset):
    """CPU token rows, with a causal first-label mask and optional variable length."""
    def __init__(self, input_ids, attention_mask, labels, *, metadata=None):
        values = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if any(not isinstance(x, torch.Tensor) or x.device.type != "cpu" for x in values.values()):
            raise ValueError("TokenDataset requires CPU tensors")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or any(x.shape != input_ids.shape for x in values.values()):
            raise ValueError("Token tensors must share a nonempty [examples, sequence] shape")
        self._init_rows([{k: v[i] for k, v in values.items()} for i in range(len(input_ids))], metadata)

    @classmethod
    def from_rows(cls, rows, *, metadata=None):
        instance = cls.__new__(cls)
        instance._init_rows(rows, metadata)
        return instance

    def _init_rows(self, rows, metadata):
        self.rows = []
        for number, row in enumerate(rows):
            if set(row) != {"input_ids", "attention_mask", "labels"}:
                raise ValueError("Token rows require input_ids, attention_mask and labels")
            raw = {k: torch.as_tensor(v) for k, v in row.items()}
            if any(v.device.type != "cpu" or v.ndim != 1 for v in raw.values()):
                raise ValueError("Token rows must be one-dimensional CPU tensors")
            ids, mask, labels = (raw[k] for k in ("input_ids", "attention_mask", "labels"))
            if len(ids) < 2 or mask.shape != ids.shape or labels.shape != ids.shape:
                raise ValueError("Every row must have matching sequence lengths of at least two")
            if ids.dtype not in (torch.int32, torch.int64) or labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("Input IDs and labels must be integer tensors")
            if bool((ids < 0).any()) or bool(((labels < 0) & labels.ne(-100)).any()):
                raise ValueError("Token IDs must be nonnegative; labels also permit -100")
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("Attention masks must be binary")
            if not bool(labels[~mask.bool()].eq(-100).all()):
                raise ValueError("Padding labels must be -100")
            values = {k: v.detach().to(torch.bool if k == "attention_mask" else torch.long).clone().contiguous()
                      for k, v in raw.items()}
            # Causal prediction never supervises the first position. Mark it
            # explicitly so public HF Trainer's item count matches shifted loss.
            values["labels"][0] = -100
            if not bool(values["labels"][1:].ne(-100).any()):
                raise ValueError(f"Row {number} has no supervised causal targets; check truncation/masking")
            self.rows.append(values)
        if not self.rows:
            raise ValueError("Dataset cannot be empty")
        self.metadata = dict(metadata or {})
        self.metadata["first_label_masked"] = True

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {k: v.clone() for k, v in self.rows[index].items()}

    @property
    def fixed_length(self):
        return len({len(r["input_ids"]) for r in self.rows}) == 1

    def tensors(self):
        if not self.fixed_length:
            raise ValueError("OptimizerPlus integration requires fixed-length inputs")
        return {k: torch.stack([row[k] for row in self.rows]) for k in self.rows[0]}

    def fingerprint(self):
        value = hashlib.sha256()
        for row in self.rows:
            for key in sorted(row):
                tensor = row[key]
                value.update(f"{key}:{list(tensor.shape)}:{tensor.dtype}".encode())
                value.update(tensor.numpy().tobytes())
        return value.hexdigest()

    def ordered(self, indices):
        return TokenDataset.from_rows((self.rows[i] for i in indices), metadata=self.metadata)

    def counts(self, start=0, stop=None):
        rows = self.rows[start:stop]
        return dict(examples=len(rows), input_tokens=sum(int(r["attention_mask"].sum()) for r in rows),
                    supervised_tokens=sum(int(r["labels"][1:].ne(-100).sum()) for r in rows))


class TokenCollator:
    def __init__(self, pad_token_id=0):
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("pad_token_id must be a nonnegative integer")
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        length = max(len(row["input_ids"]) for row in rows)
        result = {}
        for key, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.stack([torch.cat((row[key], row[key].new_full((length-len(row[key]),), fill)))
                                       for row in rows])
        return result


def load_records(source):
    """Read local JSON, JSONL or CSV; HF datasets/list-like sources also work."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open(encoding="utf-8") as stream:
            if path.suffix == ".jsonl":
                rows = [json.loads(line) for line in stream if line.strip()]
            elif path.suffix == ".json":
                rows = json.load(stream)
                if not isinstance(rows, list):
                    raise ValueError("JSON must contain an array of records")
            elif path.suffix == ".csv":
                rows = list(csv.DictReader(stream))
            else:
                raise ValueError("Use a local JSON, JSONL or CSV file")
        return rows
    if isinstance(source, dict) or not hasattr(source, "__len__"):
        raise TypeError("Supply a finite dataset/list of records; streaming and split dictionaries are not implemented")
    return source


def prepare_dataset(source, *, task, tokenizer=None, config=None):
    if task not in {"cpt", "sft"}:
        raise ValueError("task must be cpt or sft")
    if isinstance(source, TokenDataset):
        return source
    if tokenizer is None:
        raise ValueError("Raw records require a model-compatible tokenizer")
    cfg = config or DataConfig()
    if cfg.packing and task != "cpt":
        raise ValueError("Packing is currently implemented for CPT only")
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Set tokenizer.pad_token explicitly before preparation")
    eos = tokenizer.eos_token_id
    if cfg.add_eos and eos is None:
        raise ValueError("add_eos requires tokenizer.eos_token_id")
    rows = []
    truncated = 0
    stream_ids = []
    source_count = 0
    for record in load_records(source):
        source_count += 1
        if task == "cpt":
            text = record if isinstance(record, str) else record[cfg.text_field]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("CPT records require nonempty text")
            ids = list(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            labels = ids.copy()
        elif not isinstance(record, dict):
            raise TypeError("SFT records must be dictionaries")
        elif cfg.messages_field in record:
            messages = record[cfg.messages_field]
            if cfg.sft_loss not in {"auto", "assistant", "all"}:
                raise ValueError("Conversation data uses assistant or all loss")
            assistant = cfg.sft_loss != "all"
            template = tokenizer.get_chat_template()
            if assistant and not re.search(r"\{%[-+]?\s*generation\s*[-+]?%\}", template):
                raise ValueError("Assistant-only loss requires generation blocks in the tokenizer's chat template; choose all explicitly or supply a compatible template")
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_assistant_tokens_mask=assistant)
            ids = list(encoded["input_ids"])
            if assistant:
                mask = encoded.get("assistant_masks")
                if mask is None or len(mask) != len(ids):
                    raise ValueError("Tokenizer did not return a valid assistant token mask")
                labels = [token if selected else -100 for token, selected in zip(ids, mask)]
            else:
                labels = ids.copy()
        elif cfg.prompt_field in record and cfg.completion_field in record:
            if cfg.sft_loss not in {"auto", "completion", "all"}:
                raise ValueError("Prompt/completion data uses completion or all loss")
            prompt, completion = record[cfg.prompt_field], record[cfg.completion_field]
            if not isinstance(prompt, str) or not isinstance(completion, str) or not completion.strip():
                raise ValueError("Prompt/completion fields must be strings with a nonempty completion")
            if cfg.sft_loss == "all":
                ids = list(tokenizer(prompt+completion, add_special_tokens=True, truncation=False)["input_ids"])
                labels = ids.copy()
            else:
                if not getattr(tokenizer, "is_fast", False):
                    raise ValueError("Completion-only masking requires a fast tokenizer with offsets")
                encoded = tokenizer(prompt+completion, add_special_tokens=True, truncation=False,
                                    return_offsets_mapping=True)
                ids = list(encoded["input_ids"])
                # Exclude a token straddling the prompt boundary rather than
                # train on prompt text or assume separately tokenized prefixes.
                labels = [token if start >= len(prompt) and end > start else -100
                          for token, (start, end) in zip(ids, encoded["offset_mapping"])]
        else:
            raise ValueError("SFT needs messages or prompt/completion fields")
        # Chat templates own their end-of-turn tokens; do not append another EOS.
        is_chat = task == "sft" and cfg.messages_field in record
        if cfg.add_eos and not is_chat and (not ids or ids[-1] != eos):
            ids.append(eos)
            labels.append(eos)
        if cfg.packing:
            stream_ids.extend(ids)
            continue
        if len(ids) > cfg.max_length:
            if cfg.truncation == "error":
                raise ValueError("Record exceeds max_length; choose truncation='truncate' explicitly")
            truncated += 1
            ids, labels = ids[:cfg.max_length], labels[:cfg.max_length]
        rows.append(_row(ids, labels, cfg, pad))
    discarded_tail = 0
    if cfg.packing:
        for begin in range(0, len(stream_ids), cfg.max_length):
            ids = stream_ids[begin:begin+cfg.max_length]
            if len(ids) < 2:
                discarded_tail += len(ids)
                continue
            rows.append(_row(ids, ids.copy(), cfg, pad))
    return TokenDataset.from_rows(rows, metadata=dict(task=task, source_records=source_count,
        truncated_records=truncated, packing=cfg.packing, padding=cfg.padding,
        discarded_packing_tail_tokens=discarded_tail,
        tokenizer_class=type(tokenizer).__qualname__))


def _row(ids, labels, cfg, pad):
    mask = [1] * len(ids)
    if cfg.padding == "fixed":
        missing = cfg.max_length - len(ids)
        ids = ids + [pad] * missing
        labels = labels + [-100] * missing
        mask += [0] * missing
    return dict(input_ids=ids, attention_mask=mask, labels=labels)
