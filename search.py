# п.3 - чанки, ембединги, векторний і keyword пошук
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from llm import embed, load_json, save_json

CHUNKS_PATH = config.OUT_DIR / "chunks.json"
EMB_PATH = config.OUT_DIR / "embeddings.npy"

# (запит, очікуваний документ, тип)
COMPARE_QUERIES = [
    ("якого репера вбили під час пограбування", "03_xxxtentacion_death", "перефразований"),
    ("музикант втратив свідомість одразу після посадки літака", "05_juice_wrld_death", "перефразований"),
    ("родина загиблого артиста судиться з його менеджерами", "02_lil_peep_legacy", "перефразований"),
    ("XO Tour Llif3", "09_emo_rap_trend", "точна назва"),
    ("Live Free 999", "06_juice_wrld_legacy", "точна назва"),
    ("Smokeasac", "02_lil_peep_legacy", "точне ім'я"),
]


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+(?=[А-ЯІЇЄҐA-Z0-9«])", text) if s.strip()]


def chunk_document(doc_id, text, max_words=config.CHUNK_WORDS, overlap=config.CHUNK_OVERLAP_SENT):
    # речення до ~120 слів, перекриття 1 речення
    title, *body = text.strip().split("\n", 1)
    sents = split_sentences(body[0] if body else title)
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur:
        chunks.append(cur)
    # заголовок новини в кожен чанк
    return [{"id": f"{doc_id}#{i}", "doc_id": doc_id, "title": title.strip(),
             "text": f"{title.strip()}. " + " ".join(c)} for i, c in enumerate(chunks)]


def build_chunks():
    chunks = []
    for p in sorted(config.CORPUS_DIR.glob("*.txt")):
        chunks += chunk_document(p.stem, p.read_text(encoding="utf-8"))
    save_json(chunks, CHUNKS_PATH)
    lens = [len(c["text"].split()) for c in chunks]
    print(f"[chunks] {len(chunks)} чанків, слів: min={min(lens)}, avg={np.mean(lens):.0f}, max={max(lens)}")
    return chunks


class VectorStore:
    def __init__(self):
        self.chunks = load_json(CHUNKS_PATH) if CHUNKS_PATH.exists() else build_chunks()
        if EMB_PATH.exists():
            self.emb = np.load(EMB_PATH)
        else:
            print(f"[embed] {len(self.chunks)} чанків через {config.GEMINI_EMBED_MODEL} ...")
            self.emb = np.array(embed([c["text"] for c in self.chunks]), dtype=np.float32)
            np.save(EMB_PATH, self.emb)
        self.emb = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)

    def search(self, query, k=config.TOP_N):
        q = np.array(embed([query], is_query=True)[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = self.emb @ q  # cosine, вектори нормовані
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]


class KeywordSearch:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[\w-]+\b")
        self.m = self.vec.fit_transform([c["text"] for c in chunks])

    def tfidf(self, query, k=config.TOP_N):
        sims = cosine_similarity(self.vec.transform([query]), self.m)[0]
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx if sims[i] > 0]

    def substring(self, query, k=config.TOP_N):
        q = query.lower()
        return [(c, 1.0) for c in self.chunks if q in c["text"].lower()][:k]


def rank_of(results, doc_id):
    for i, (c, _) in enumerate(results, 1):
        if c["doc_id"] == doc_id:
            return i
    return None


def compare():
    vs = VectorStore()
    kw = KeywordSearch(vs.chunks)
    md = ["# Порівняння keyword-пошуку і векторного пошуку\n",
          f"Ембединги: `{config.GEMINI_EMBED_MODEL}`, top-{config.TOP_N}. "
          "Ранг — позиція першого чанка з очікуваного документа (— = не знайдено).\n",
          "| Запит | Тип | Очікуваний документ | Підрядок | TF-IDF | Вектор |", "|---|---|---|---|---|---|"]
    details = []
    for q, expected, kind in COMPARE_QUERIES:
        sub, tf, vec = kw.substring(q), kw.tfidf(q), vs.search(q)
        r = [rank_of(x, expected) for x in (sub, tf, vec)]
        md.append(f"| {q} | {kind} | {expected} | " + " | ".join(str(x) if x else "—" for x in r) + " |")
        details.append(f"\n### «{q}»\n")
        for name, res in (("TF-IDF", tf), ("Вектор", vec)):
            details.append(f"**{name}:**\n")
            details += [f"{i}. `{c['id']}` ({s:.3f}) — {c['text'][len(c['title']) + 2:][:140]}…"
                        for i, (c, s) in enumerate(res[:3], 1)] or ["(нічого)"]
            details.append("")
    out = "\n".join(md + ["\n## Топ-3 по кожному запиту"] + details)
    (config.OUT_DIR / "search_comparison.md").write_text(out, encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "compare":
        for c, s in VectorStore().search(" ".join(sys.argv[1:])):
            print(f"{s:.3f}  {c['id']}  {c['text'][:150]}…")
    else:
        compare()
