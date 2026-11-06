import csv
import random
import sys
from collections import Counter

import config
from llm import load_json

CSV_PATH = config.OUT_DIR / "verification.csv"
ERROR_TYPES = "omission | hallucination | wrong_link | wrong_type | wrong_direction | wrong_year"


def sample(n=20, seed=42):
    facts = load_json(config.OUT_DIR / "facts.json")
    docs = {p.stem: p.read_text(encoding="utf-8") for p in config.CORPUS_DIR.glob("*.txt")}
    random.seed(seed)
    picked = random.sample(facts, min(n, len(facts)))
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "doc_id", "subject", "predicate", "object", "year", "evidence",
                    "evidence_in_text", "correct", "error_type", "comment"])
        for x in picked:
            # чи є цитата в тексті
            ev_ok = "так" if x["evidence"].strip()[:60] in docs.get(x["doc_id"], "") else "НІ"
            w.writerow([x["id"], x["doc_id"], x["subject"], x["predicate"], x["object"], x["year"],
                        x["evidence"], ev_ok, "", "", ""])
    print(f"Збережено {len(picked)} фактів у {CSV_PATH}")
    print("Відкрийте у Excel, звірте кожен факт з текстом і заповніть:")
    print("  correct = 1/0;  error_type: " + ERROR_TYPES)


def score():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig"), delimiter=";"))
    filled = [r for r in rows if r["correct"].strip() in ("0", "1")]
    if not filled:
        sys.exit("Стовпець correct не заповнено")
    ok = sum(r["correct"].strip() == "1" for r in filled)
    errs = Counter(r["error_type"].strip() or "?" for r in filled if r["correct"].strip() == "0")
    lines = [f"# Ручна перевірка фактів\n",
             f"Перевірено: {len(filled)}, коректних: {ok} ({ok / len(filled):.0%})\n",
             "| Тип помилки | Кількість |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in errs.most_common()]
    lines += ["\n## Некоректні факти\n"]
    lines += [f"- {r['id']}: ({r['subject']}) -[{r['predicate']}]-> ({r['object']}) — "
              f"{r['error_type']}; {r['comment']}" for r in filled if r["correct"].strip() == "0"]
    out = "\n".join(lines)
    (config.OUT_DIR / "verification_result.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
