import torch
from typing import List, Dict, Any

IGNORE_INDEX = -100


class DataCollatorForChatSFT:
    def __init__(self, tokenizer, max_length=4096, padding="longest"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __call__(self, features):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feature in features:
            messages = feature["messages"]

            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            prompt_text = self.tokenizer.apply_chat_template(
                messages[:-1]
                + [{"role": "assistant", "content": ""}],
                tokenize=False,
            )

            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]

            encoded_prompt = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

            prompt_len = encoded_prompt["input_ids"].shape[-1]

            labels = input_ids.clone()
            labels[:prompt_len] = IGNORE_INDEX

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        batch = self.tokenizer.pad(
            {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels,
            },
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return batch
