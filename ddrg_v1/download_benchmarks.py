import argparse
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

import pandas as pd
from datasets import load_dataset


LOGIQA_TEST_URL = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt"
MATHQA_ZIP_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
AIW_PROMPTS_URL = "https://raw.githubusercontent.com/LAION-AI/AIW/main/prompts/prompts.json"

LETTER_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def benchmarks_root() -> Path:
    return project_root() / "benchmarks"


def hf_cache_dir() -> Path:
    path = project_root() / ".hf-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_bytes(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def option_label(index: int) -> str:
    return LETTER_CHOICES[index]


def format_options(options: Iterable[str]) -> str:
    lines = [f"{option_label(idx)}. {normalize_space(str(option))}" for idx, option in enumerate(options)]
    return "\n".join(lines)


def format_multiple_choice_question(stem: str, options: Iterable[str]) -> str:
    stem_text = str(stem).strip()
    options_text = format_options(options)
    return (
        f"{stem_text}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer with the option letter only."
    )


def dedupe_rows(df: pd.DataFrame, question_col: str, answer_col: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[question_col] = cleaned[question_col].astype(str).str.strip()
    cleaned[answer_col] = cleaned[answer_col].astype(str).str.strip()
    cleaned = cleaned.dropna(subset=[question_col, answer_col])
    cleaned = cleaned[(cleaned[question_col] != "") & (cleaned[answer_col] != "")]
    cleaned = cleaned.drop_duplicates(subset=[question_col, answer_col]).reset_index(drop=True)
    return cleaned


def _logiqa_process_answer(answer: str) -> str:
    if not any(answer.startswith(x) for x in "ABCD"):
        return answer
    return answer[3:]


def _logiqa_process_sentences(text: str) -> str:
    text = text.replace("\n", "")
    sents = text.split(".")
    processed = ""
    for sent in sents:
        if not sent:
            continue
        if not processed:
            processed += sent
        elif sent[0].isnumeric():
            processed += "." + sent
        else:
            processed += ". " + sent
    processed = processed.replace("  ", " ")
    processed = processed.replace("\\'", "'")
    while processed.endswith(" "):
        processed = processed[:-1]
    if re.match(r"^[A-Z][\w\s]+[?.!]$", processed) is None:
        processed += "."
    processed = processed.replace("?.", "?")
    processed = processed.replace("!.", "!")
    processed = processed.replace("..", ".")
    return processed


def build_logiqa() -> pd.DataFrame:
    raw_text = download_bytes(LOGIQA_TEST_URL).decode("utf-8")
    lines = [_logiqa_process_sentences(line) for line in raw_text.splitlines()]
    rows = []
    for sample_idx in range(len(lines) // 8):
        row = sample_idx * 8
        correct_answer = lines[row + 1].replace(".", "").strip().upper()
        context = lines[row + 2].strip()
        query = lines[row + 3].strip()
        answers = [_logiqa_process_answer(lines[row + 4 + offset]).strip() for offset in range(4)]
        question = (
            f"Passage:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Options:\n{format_options(answers)}\n\n"
            "Answer with the option letter only."
        )
        rows.append({"Question": question, "Label": correct_answer})
    return dedupe_rows(pd.DataFrame(rows), "Question", "Label")


def build_lsat_ar() -> pd.DataFrame:
    dataset = load_dataset(
        "dmayhem93/agieval-lsat-ar",
        split="test",
        cache_dir=str(hf_cache_dir()),
    )
    rows = []
    for item in dataset:
        gold = item["gold"]
        gold_index = gold[0] if isinstance(gold, list) else gold
        question = str(item["query"]).strip()
        if not question.endswith("Answer with the option letter only."):
            question = f"{question}\nAnswer with the option letter only."
        rows.append({"Question": question, "Label": option_label(int(gold_index))})
    return dedupe_rows(pd.DataFrame(rows), "Question", "Label")


def build_mathqa() -> pd.DataFrame:
    archive = zipfile.ZipFile(BytesIO(download_bytes(MATHQA_ZIP_URL)))
    test_rows = json.loads(archive.read("test.json").decode("utf-8"))
    rows = []
    for item in test_rows:
        problem = str(item["Problem"]).strip()
        options = normalize_space(str(item["options"]))
        correct = str(item["correct"]).strip().upper()
        question = f"{problem}\n\nOptions:\n{options}\n\nAnswer with the option letter only."
        rows.append({"Question": question, "Label": correct})
    return dedupe_rows(pd.DataFrame(rows), "Question", "Label")


def build_medqa() -> pd.DataFrame:
    dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options-hf",
        split="test",
        cache_dir=str(hf_cache_dir()),
    )
    rows = []
    for item in dataset:
        stem = str(item["sent1"]).strip()
        extra = str(item["sent2"]).strip()
        if extra:
            stem = f"{stem}\n\n{extra}"
        options = [item[f"ending{idx}"] for idx in range(4)]
        rows.append(
            {
                "Question": format_multiple_choice_question(stem, options),
                "Label": option_label(int(item["label"])),
            }
        )
    return dedupe_rows(pd.DataFrame(rows), "Question", "Label")


def _coerce_prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return "\n".join(parts)
    if value is None:
        return ""
    return str(value).strip()


def build_aiw_variants() -> tuple[pd.DataFrame, pd.DataFrame]:
    prompts = json.loads(download_bytes(AIW_PROMPTS_URL).decode("utf-8"))
    easy_rows = []
    hard_rows = []
    for item in prompts:
        description = str(item.get("description", "") or "").strip()
        description_upper = description.upper()
        prompt = _coerce_prompt_text(item.get("prompt"))
        answer = _coerce_prompt_text(item.get("right_answer"))
        if not prompt or not answer:
            continue
        if "DO NOT USE" in prompt.upper():
            continue
        row = {"questions": prompt, "answers": answer}
        if description_upper.startswith("AIW++"):
            continue
        if description_upper.startswith("AIW+"):
            hard_rows.append(row)
        else:
            easy_rows.append(row)
    easy_df = dedupe_rows(pd.DataFrame(easy_rows), "questions", "answers")
    hard_df = dedupe_rows(pd.DataFrame(hard_rows), "questions", "answers")
    return easy_df, hard_df


def write_outputs(root: Path) -> dict[str, Path]:
    outputs = {
        "logiqa": root / "logiqa" / "logiqa_test.csv",
        "lsat_ar": root / "lsat_ar" / "lsat-ar.csv",
        "mathqa": root / "mathqa" / "mathqa.csv",
        "medqa": root / "medqa" / "medqa.csv",
        "aiw": root / "aiw" / "AIW_easy.pkl",
        "aiw+": root / "aiw+" / "AIW_hard.pkl",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    logiqa_df = build_logiqa()
    lsat_df = build_lsat_ar()
    mathqa_df = build_mathqa()
    medqa_df = build_medqa()
    aiw_easy_df, aiw_hard_df = build_aiw_variants()

    logiqa_df.to_csv(outputs["logiqa"], index=False)
    lsat_df.to_csv(outputs["lsat_ar"], index=False)
    mathqa_df.to_csv(outputs["mathqa"], index=False)
    medqa_df.to_csv(outputs["medqa"], index=False)
    aiw_easy_df.to_pickle(outputs["aiw"])
    aiw_hard_df.to_pickle(outputs["aiw+"])

    print(f"logiqa: {len(logiqa_df)} rows -> {outputs['logiqa']}")
    print(f"lsat_ar: {len(lsat_df)} rows -> {outputs['lsat_ar']}")
    print(f"mathqa: {len(mathqa_df)} rows -> {outputs['mathqa']}")
    print(f"medqa: {len(medqa_df)} rows -> {outputs['medqa']}")
    print(f"aiw: {len(aiw_easy_df)} rows -> {outputs['aiw']}")
    print(f"aiw+: {len(aiw_hard_df)} rows -> {outputs['aiw+']}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert DDRG benchmark datasets.")
    parser.add_argument(
        "--benchmarks-root",
        default=str(benchmarks_root()),
        help="Directory where the converted benchmark files should be written.",
    )
    args = parser.parse_args()
    write_outputs(Path(args.benchmarks_root).resolve())


if __name__ == "__main__":
    main()
