import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question using only the provided document chunks. "
    "Each factual statement must be supported by a citation. "
    "Use [number] to cite the document chunk."
)

MAX_CITATIONS_PER_STATEMENT = 10
AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"


class AutoAIS:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            AUTOAIS_MODEL, use_fast=False
        )
        self.entailment_id = 209 

    @torch.inference_mode()
    def entails(self, premise: str, hypothesis: str) -> bool:
        text = f"premise: {premise} hypothesis: {hypothesis}"
        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(self.model.device)

        out = self.model.generate(
            input_ids,
            output_scores=True,
            return_dict_in_generate=True,
        ).scores[0]

        prob = torch.softmax(out[0], -1)[self.entailment_id].item()
        return prob >= 0.5


def format_citations(ids: List[int]) -> str:
    return "".join(f"[{i}]" for i in ids)


def build_user_prompt(question: str, chunks: List[Dict]) -> str:
    lines = [f"[{c['chunk_id']}] {c['text']}" for c in chunks]
    return f"Question: {question}\n\nDocuments:\n" + "\n".join(lines)


def sample_docs(docs: List[Dict], k: int = 10) -> List[Dict]:
    docs = [d for d in docs if d.get("text")]
    random.shuffle(docs)
    return docs[:k]


def build_chunks(docs: List[Dict]) -> List[Dict]:
    return [
        {"chunk_id": i + 1, "text": d["text"].strip(), "title": d.get("title", "")}
        for i, d in enumerate(docs)
    ]


def process_qampari(item: Dict, autoais: AutoAIS) -> List[Dict]:
    question = item.get("question")
    answers_raw = item.get("answers", [])
    docs = item.get("docs", [])

    if not question or not answers_raw or not docs:
        return []

    answers = [
        " ".join(a).strip()
        for a in answers_raw
        if isinstance(a, list) and " ".join(a).strip()
    ]
    if not answers:
        return []

    sampled_docs = sample_docs(docs, k=10)
    chunks = build_chunks(sampled_docs)

    answer_spans = []

    for answer in answers:
        claim = f"{question} {answer}"

        supporting = []
        for c in chunks:
            premise = f"Title: {c['title']}\n{c['text']}"
            if autoais.entails(premise, claim):
                supporting.append(c["chunk_id"])

        if not supporting:
            continue

        answer_spans.append(
            f"{answer} {format_citations(supporting[:MAX_CITATIONS_PER_STATEMENT])}"
        )

    if not answer_spans:
        return []

    assistant = ", ".join(answer_spans)

    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, chunks)},
                {"role": "assistant", "content": assistant},
            ]
        }
    ]



def process_eli5(item: Dict, autoais: AutoAIS) -> List[Dict]:
    question = item.get("question")
    docs = item.get("docs", [])
    answers = item.get("answers", [])

    if not question or not docs or not answers:
        return []

    answer_text = answers[0] if isinstance(answers, list) else answers
    sentences = [s.strip() for s in sent_tokenize(answer_text) if s.strip()]

    sampled_docs = sample_docs(docs, k=10)
    chunks = build_chunks(sampled_docs)

    sent_spans = []

    for sent in sentences:
        supporting = []
        for c in chunks:
            premise = f"Title: {c['title']}\n{c['text']}"
            if autoais.entails(premise, sent):
                supporting.append(c["chunk_id"])

        if not supporting:
            continue

        sent_spans.append(
            f"{sent} {format_citations(supporting[:MAX_CITATIONS_PER_STATEMENT])}"
        )

    if not sent_spans:
        return []

    assistant = " ".join(sent_spans)

    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, chunks)},
                {"role": "assistant", "content": assistant},
            ]
        }
    ]



def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    autoais = AutoAIS()
    all_examples = []

    for path in args.input_files:
        data = load_json(Path(path))
        for item in tqdm(data, desc=f"Processing {path}"):
            if "answers" in item and isinstance(item["answers"][0], list):
                all_examples.extend(process_qampari(item, autoais))
            else:
                all_examples.extend(process_eli5(item, autoais))

    random.shuffle(all_examples)
    all_examples = all_examples[: args.max_samples]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_examples)} examples to {out}")

if __name__ == "__main__":
    main()