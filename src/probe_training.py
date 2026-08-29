"""Model-agnostic extraction and training utilities for demographic probes."""

from __future__ import annotations

import csv
import json
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe. Please ensure that your responses are socially "
    "unbiased and positive in nature. If a question does not make any sense, or is "
    "not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)

ATTRIBUTE_SPECS = {
    "gender": {
        "marker": "gender",
        "prompt_name": "gender",
        "classes": ["male", "female"],
    },
    "age": {
        "marker": "age",
        "prompt_name": "age",
        "classes": ["child", "adolescent", "adult", "older adult"],
    },
    "education": {
        "marker": "education",
        "prompt_name": "education level",
        "classes": ["someschool", "highschool", "collegemore"],
    },
    "socioeconomic": {
        "marker": "socioeco",
        "prompt_name": "socioeconomic status",
        "classes": ["low", "middle", "high"],
    },
}


@dataclass(frozen=True)
class ConversationExample:
    source: str
    label: int
    class_name: str
    messages: list[dict[str, str]]


class LinearSigmoidProbe:
    """TalkTuner-compatible linear one-vs-rest probe."""

    def __new__(cls, input_dim: int, num_classes: int):
        import torch.nn as nn

        class _Probe(nn.Module):
            def __init__(self):
                super().__init__()
                # Keep this module name so checkpoints use proj.0.weight/proj.0.bias.
                self.proj = nn.Sequential(nn.Linear(input_dim, num_classes), nn.Sigmoid())
                nn.init.normal_(self.proj[0].weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.proj[0].bias)

            def forward(self, activations):
                return self.proj(activations)

        return _Probe()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable splits and probe training."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_conversation(text: str) -> list[dict[str, str]]:
    """Convert the HUMAN/ASSISTANT variants in TalkTuner data to chat messages."""
    marker = re.compile(r"^\s*(?:###\s*)?(Human|User|Assistant):\s*(.*)$", re.I)
    messages: list[dict[str, str]] = []
    role = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal role, parts
        content = "\n".join(parts).strip()
        if role and content:
            messages.append({"role": role, "content": content})
        role, parts = None, []

    for line in text.splitlines():
        match = marker.match(line)
        if match:
            flush()
            role = "assistant" if match.group(1).lower() == "assistant" else "user"
            parts = [match.group(2)]
        elif role is not None:
            parts.append(line)
    flush()

    # Reading and controlling probes predict before the final assistant response.
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
    while messages and messages[-1]["role"] != "user":
        messages.pop()
    return messages


def load_examples(dataset_dir: Path, attribute: str) -> tuple[list[ConversationExample], dict]:
    """Read matching TalkTuner conversations directly from compressed archives."""
    spec = ATTRIBUTE_SPECS[attribute]
    label_to_id = {name: index for index, name in enumerate(spec["classes"])}
    filename_pattern = re.compile(
        rf"_{re.escape(spec['marker'])}_([^/]+)\.txt$", re.I
    )
    examples: list[ConversationExample] = []
    archive_stats = {}

    archives = sorted(dataset_dir.glob(f"*{spec['marker']}*.zip"))
    # llama_gender_3.zip is retained for provenance, although its entries are
    # misnamed with _age_ and are therefore intentionally rejected below.
    for archive in archives:
        accepted = 0
        rejected = 0
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                match = filename_pattern.search(entry.filename)
                if not match or match.group(1) not in label_to_id:
                    rejected += 1
                    continue
                text = zipped.read(entry).decode("utf-8")
                messages = parse_conversation(text)
                if not messages:
                    rejected += 1
                    continue
                class_name = match.group(1)
                examples.append(
                    ConversationExample(
                        source=f"{archive.name}::{entry.filename}",
                        label=label_to_id[class_name],
                        class_name=class_name,
                        messages=messages,
                    )
                )
                accepted += 1
        archive_stats[archive.name] = {"accepted": accepted, "rejected": rejected}

    if not examples:
        raise ValueError(f"No valid {attribute} examples found in {dataset_dir}")
    return examples, archive_stats


def render_probe_prompt(tokenizer, messages, attribute: str, channel: str) -> str:
    """Render the same reading/control endpoints used by TalkTuner for any chat model."""
    chat = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *messages]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    if channel == "reading":
        prompt += f"I think the {ATTRIBUTE_SPECS[attribute]['prompt_name']} of this user is"
    return prompt


def load_causal_model(model_name: str, device: str, dtype: str):
    """Load a Hugging Face causal LM without architecture-specific dimensions."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype_map[dtype],
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def extract_activations(
    *,
    model,
    tokenizer,
    examples: list[ConversationExample],
    attribute: str,
    channel: str,
    device: str,
    batch_size: int,
    max_length: int,
    output_path: Path,
    model_name: str,
    archive_stats: dict,
) -> None:
    """Cache the final-token residual stream for every model layer."""
    import torch

    prompts = [render_probe_prompt(tokenizer, ex.messages, attribute, channel) for ex in examples]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
            final_positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(final_positions.shape[0], device=device)
            # hidden_states[0] is the embedding output, matching TalkTuner layer 0.
            selected = torch.stack(
                [state[batch_indices, final_positions] for state in outputs.hidden_states], dim=1
            )
            chunks.append(selected.to(dtype=torch.float16, device="cpu"))
            done = min(start + batch_size, len(prompts))
            print(f"[extract] {attribute}/{channel}: {done}/{len(prompts)}", flush=True)

    payload = {
        "activations": torch.cat(chunks),
        "labels": torch.tensor([ex.label for ex in examples], dtype=torch.long),
        "class_names": ATTRIBUTE_SPECS[attribute]["classes"],
        "sources": [ex.source for ex in examples],
        "attribute": attribute,
        "channel": channel,
        "model_name": model_name,
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "max_length": max_length,
        "archive_stats": archive_stats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[ok] cached {tuple(payload['activations'].shape)} at {output_path}")


def stratified_indices(labels, validation_fraction: float, seed: int):
    from sklearn.model_selection import train_test_split

    indices = list(range(len(labels)))
    return train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels.tolist(),
    )


def train_probes_from_cache(
    *,
    cache_path: Path,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
) -> list[dict]:
    """Train and save one TalkTuner-compatible classifier per cached layer."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    payload = torch.load(cache_path, map_location="cpu")
    activations = payload["activations"]
    labels = payload["labels"]
    class_names = payload["class_names"]
    train_indices, val_indices = stratified_indices(labels, validation_fraction, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    for layer in range(activations.shape[1]):
        set_seed(seed + layer)
        train_data = TensorDataset(
            activations[train_indices, layer].float(), labels[train_indices]
        )
        val_data = TensorDataset(activations[val_indices, layer].float(), labels[val_indices])
        generator = torch.Generator().manual_seed(seed + layer)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, generator=generator
        )
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        probe = LinearSigmoidProbe(activations.shape[2], len(class_names)).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        best_accuracy = -1.0
        best_epoch = 0
        checkpoint = output_dir / f"{payload['attribute']}_probe_at_layer_{layer}.pth"

        for epoch in range(1, epochs + 1):
            probe.train()
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                one_hot = torch.nn.functional.one_hot(
                    targets, num_classes=len(class_names)
                ).float()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(probe(features), one_hot)
                loss.backward()
                optimizer.step()

            probe.eval()
            correct = total = 0
            validation_loss = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    one_hot = torch.nn.functional.one_hot(
                        targets, num_classes=len(class_names)
                    ).float()
                    probabilities = probe(features)
                    validation_loss += criterion(probabilities, one_hot).item() * len(targets)
                    correct += (probabilities.argmax(dim=1) == targets).sum().item()
                    total += len(targets)
            accuracy = correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                torch.save(probe.state_dict(), checkpoint)

        record = {
            "attribute": payload["attribute"],
            "channel": payload["channel"],
            "layer": layer,
            "validation_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "train_examples": len(train_indices),
            "validation_examples": len(val_indices),
            "checkpoint": checkpoint.name,
        }
        metrics.append(record)
        print(
            f"[train] {payload['attribute']}/{payload['channel']} layer={layer} "
            f"accuracy={best_accuracy:.4f} epoch={best_epoch}",
            flush=True,
        )

    metrics_path = output_dir / f"{payload['attribute']}_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    metadata = {
        key: payload[key]
        for key in (
            "attribute", "channel", "model_name", "hidden_size",
            "num_hidden_layers", "class_names", "max_length", "archive_stats"
        )
    }
    metadata.update(
        {
            "checkpoint_layers": list(range(activations.shape[1])),
            "training": {
                "objective": "sigmoid + binary cross entropy on one-hot labels",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_fraction": validation_fraction,
                "seed": seed,
            },
        }
    )
    (output_dir / f"{payload['attribute']}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metrics
