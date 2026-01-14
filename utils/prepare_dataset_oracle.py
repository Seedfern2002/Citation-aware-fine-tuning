import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question using only the provided document chunks. "
    "Each factual statement must be supported by a citation. "
    "Use [number] to cite the document chunk."
)

MAX_CITATIONS_PER_STATEMENT = 10


def format_citations(citation_ids_1based: List[int]) -> str:
    return "".join(f"[{i}]" for i in citation_ids_1based)


def build_user_prompt(question: str, chunks: List[Dict]) -> str:
    lines = [f"[{c['chunk_id']}] {c['text']}" for c in chunks]
    return f"Question: {question}\n\nDocuments:\n" + "\n".join(lines)


def build_chunks_from_docs(docs: List[Dict]) -> List[Dict]:
    chunks = []
    for i, doc in enumerate(docs):
        text = doc.get("text", "").strip()
        assert len(text.split()) <= 100, f"Document {i + 1} exceeds 100 words"
        if not text:
            continue
        chunks.append(
            {
                "chunk_id": i + 1,  
                "text": text,
            }
        )
    return chunks


def choose_citations(supporting_doc_indices: List[int]) -> List[int]:
    citation_ids = [i + 1 for i in supporting_doc_indices]
    return citation_ids[:MAX_CITATIONS_PER_STATEMENT]


def process_qampari(ora: Dict) -> List[Dict]:
    outputs = []

    question = ora.get("question", "")
    answers_raw = ora.get("answers", [])
    docs = ora.get("docs", [])

    if not question or not answers_raw or not docs:
        return outputs

    answers = [
        " ".join(a).strip()
        for a in answers_raw
        if isinstance(a, list) and " ".join(a).strip()
    ]
    if not answers:
        return outputs

    chunks = build_chunks_from_docs(docs)

    answer_spans = []

    for ans_idx, answer in enumerate(answers):
        supporting_doc_indices = []

        for doc_idx, d in enumerate(docs):
            af = d.get("answers_found")
            if not isinstance(af, list):
                continue

            if ans_idx < len(af) and af[ans_idx]:
                supporting_doc_indices.append(doc_idx)

        if not supporting_doc_indices:
            continue

        citation_ids = choose_citations(supporting_doc_indices)
        if not citation_ids:
            continue

        answer_spans.append(
            f"{answer} {format_citations(citation_ids)}"
        )

    if not answer_spans:
        return outputs

    assistant = ", ".join(answer_spans)

    outputs.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, chunks)},
                {"role": "assistant", "content": assistant},
            ]
        }
    )

    return outputs


def process_eli5(ora: Dict) -> List[Dict]:
    outputs = []

    question = ora.get("question", "")
    claims = ora.get("claims", [])
    docs = ora.get("docs", [])

    if not question or not claims or not docs:
        return outputs

    claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    if not claims:
        return outputs

    chunks = build_chunks_from_docs(docs)

    for claim_idx, claim in enumerate(claims):
        supporting_doc_indices = []

        for doc_idx, d in enumerate(docs):
            af = d.get("claims_found")
            if not isinstance(af, list):
                continue

            if claim_idx < len(af) and af[claim_idx]:
                supporting_doc_indices.append(doc_idx)

        if not supporting_doc_indices:
            continue

        citation_ids = choose_citations(supporting_doc_indices)
        if not citation_ids:
            continue

        assistant = f"{claim} {format_citations(citation_ids)}"

        outputs.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, chunks)},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )

    return outputs


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="List of *_reranked_oracle.json files",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    all_examples = []

    for file_path in args.input_files:
        ora_path = Path(file_path)
        ora_data = load_json(ora_path)

        for ora in tqdm(
            ora_data,
            desc=f"Processing {ora_path.name}",
        ):
            if "answers" in ora:
                all_examples.extend(process_qampari(ora))
            elif "claims" in ora:
                all_examples.extend(process_eli5(ora))

    random.shuffle(all_examples)
    all_examples = all_examples[: args.max_samples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_examples)} SFT examples to {out_path}")


if __name__ == "__main__":
    main()
