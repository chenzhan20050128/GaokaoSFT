#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python qLora.py \
  --base_model_path "/path/to/qwen_math_grpo_merged" \
  --dataset_path "archive/training_final.json" \
  --output_dir "outputs/qwen_math_sft" \
  --learning_rate 2e-4 \
  --num_train_epochs 2
"""
"""qLoRA SFT training script built on top of Unsloth FastLanguageModel.

This script fine-tunes a previously GRPO-adapted Qwen base using all
"解答题" samples from the provided JSON dataset. The training pipeline
leverages Hugging Face's `trl.SFTTrainer` together with Unsloth's highly
optimized qLoRA utilities.

Typical usage (PowerShell):

```
python .\qLora.py \
    --base_model_path "C:/Users/Administrator/Downloads/qwen_math_grpo_merged" \
    --dataset_path "archive/training_final.json" \
    --output_dir "outputs/qwen_math_sft" \
    --learning_rate 2e-4 \
    --num_train_epochs 2
```

Key design choices:
- **Data filtering**: retain samples whose `question_type` is exactly "解答题".
- **Formatting**: adopt ChatML-style prompt with dedicated `<think>` and
  `<final>` blocks so the model can internalise the reasoning trace before
  emitting the final answer.
- **Training configuration**: qLoRA (4-bit base, rank 64) with cosine LR
  schedule, warmup ratio, gradient checkpointing, and rich progress bars.
"""

from __future__ import annotations
                  
import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

try:
    from unsloth import FastLanguageModel
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Unsloth is required for this script. Install via `pip install unsloth`."
    ) from exc


LOGGER = logging.getLogger("qLora")
CHATML_SYSTEM_PROMPT = (
    "你是一名擅长中国高中数学解答题的教师，会先在 <think> 中给出严谨、分段的思考，"
    "再在 <final> 中给出简洁的最终答案。保持符号规范，必要时使用 LaTeX。"
)


@dataclass(slots=True)
class DataExample:
    """Simple container holding the formatted training text and metadata."""

    text: str
    difficulty: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_examples(path: Path) -> List[DataExample]:
    """Load JSON list, keep 解答题 rows, and format into ChatML transcripts."""

    LOGGER.info("Loading dataset from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        raw_rows = json.load(fh)

    examples: List[DataExample] = []
    dropped = 0
    for row in raw_rows:
        if row.get("question_type") != "解答题":
            continue

        question = (row.get("question_stem") or "").strip()
        answer = (row.get("answer") or row.get("solution") or "").strip()
        thinking = (row.get("thinking") or row.get("reasoning") or "").strip()

        if not question or not answer or not thinking:
            dropped += 1
            continue

        difficulty = (row.get("difficulty") or "").strip() or None
        formatted = format_chatml_turn(question, thinking, answer)
        examples.append(DataExample(formatted, difficulty))

    LOGGER.info(
        "Prepared %d examples (dropped %d incomplete rows)",
        len(examples),
        dropped,
    )
    return examples


def format_chatml_turn(question: str, thinking: str, answer: str) -> str:
    """Format a single instruction-response pair using ChatML-like tags."""

    return (
        "<|im_start|>system\n"
        f"{CHATML_SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        f"{thinking.strip()}\n"
        "</think>\n"
        "<final>\n"
        f"{answer.strip()}\n"
        "</final>\n"
        "<|im_end|>"
    )


def split_train_eval(
    data: Sequence[DataExample],
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Shuffle deterministically, then split into train & eval datasets."""

    if not 0.0 < eval_ratio < 0.5:
        raise ValueError("eval_ratio should be between 0 and 0.5 for stability")

    generator = random.Random(seed)
    shuffled = list(data)
    generator.shuffle(shuffled)

    eval_count = max(1, int(len(shuffled) * eval_ratio))
    train_examples = shuffled[eval_count:]
    eval_examples = shuffled[:eval_count]

    train_dataset = Dataset.from_list([{"text": ex.text} for ex in train_examples])
    eval_dataset = Dataset.from_list([{"text": ex.text} for ex in eval_examples])

    return train_dataset, eval_dataset


def print_difficulty_stats(examples: Iterable[DataExample]) -> None:
    counter: dict[str, int] = {}
    total = 0
    for ex in examples:
        key = ex.difficulty or "(未知)"
        counter[key] = counter.get(key, 0) + 1
        total += 1

    LOGGER.info("Difficulty distribution over %d samples:", total)
    for diff, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        LOGGER.info("  %-12s : %6d", diff, count)


def build_model_and_tokenizer(
    base_model_path: str,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str] | None,
) -> tuple[torch.nn.Module, "PreTrainedTokenizerFast"]:
    """Load the 4-bit base model and wrap it with a qLoRA adapter."""

    LOGGER.info("Loading base model from %s", base_model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect best precision (bf16 if available)
        load_in_4bit=True,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if target_modules is None:
        target_modules = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    LOGGER.info(
        "Applying qLoRA: r=%d, alpha=%d, dropout=%.4f, targets=%s",
        lora_r,
        lora_alpha,
        lora_dropout,
        ",".join(target_modules),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def construct_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    """Create Hugging Face TrainingArguments with rich progress output."""

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    return TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        tf32=True,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        run_name=args.run_name,
        disable_tqdm=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train qLoRA SFT with Unsloth")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="C:/Users/Administrator/Downloads/qwen_math_grpo_merged",
        help="Path to the LoRA+GRPO adapted base model to start from.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="archive/training_final.json",
        help="JSON file holding raw Gaokao samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen_math_sft",
        help="Where to store adapter checkpoints and logs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.02,
        help="Fraction of data reserved for evaluation (0-0.5).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenisation.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Per-device micro batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
        help="Per-device micro batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps (simulated batch size).",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Peak learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay coefficient."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=2.0,
        help="Number of full passes through the dataset.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override total training steps. Keep -1 to rely on epochs.",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.03, help="Warmup ratio for LR."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every N gradient updates.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps (set <=0 to disable eval during training).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Logging integrations (comma-separated). Use 'none' to disable.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="qwen-math-qlora-sft",
        help="Tracking run name for experiment managers.",
    )
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank (r).")
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha hyperparameter."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate."
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="",
        help=(
            "Comma-separated list of target modules for LoRA. Leave empty to "
            "use a sensible default covering QKV and MLP projections."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional checkpoint directory to resume training from.",
    )

    parsed = parser.parse_args()
    parsed.output_dir = Path(parsed.output_dir)
    parsed.output_dir.mkdir(parents=True, exist_ok=True)

    if parsed.report_to.lower() in {"none", "off", "disable"}:
        parsed.report_to = []
    else:
        parsed.report_to = [
            part.strip() for part in parsed.report_to.split(",") if part.strip()
        ]

    parsed.dataset_path = Path(parsed.dataset_path)
    parsed.base_model_path = str(parsed.base_model_path)

    return parsed


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    set_seed(args.seed)

    examples = load_dataset_examples(args.dataset_path)
    if not examples:
        raise SystemExit("No usable 解答题 samples found in the dataset.")

    print_difficulty_stats(examples)
    train_dataset, eval_dataset = split_train_eval(examples, args.eval_ratio, args.seed)

    model, tokenizer = build_model_and_tokenizer(
        base_model_path=args.base_model_path,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",") if args.target_modules else None,
    )

    model.config.use_cache = False  # Hugging Face Trainer compatibility

    training_args = construct_training_arguments(args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    LOGGER.info(
        "Starting training with %d train / %d eval samples",
        len(train_dataset),
        len(eval_dataset),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    LOGGER.info("Training complete: %s", train_result.metrics)

    LOGGER.info("Saving adapter to %s", args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save final merged model if needed. Users can run `unsloth.merge_lora` later.
    LOGGER.info("All done. To merge adapters, use unsloth.merge_lora when ready.")


if __name__ == "__main__":
    main()
