import os
import json
from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from training.collator import DataCollatorForChatSFT


class ChatSFTDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.samples[idx]


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--run_name", type=str, default="qwen3-alce-sft")

    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    model.config.use_cache = False

    train_dataset = ChatSFTDataset(args.train_file)

    data_collator = DataCollatorForChatSFT(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,

        bf16=args.bf16,
        fp16=not args.bf16,

        dataloader_num_workers=2,
        remove_unused_columns=False,

        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name=args.run_name,

        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
