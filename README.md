# ЛР №02 — LLM-екстракція знань, векторний пошук та GraphRAG

Предметна область: **емо-реп** (продовження ЛР-01) — корпус новин про Lil Peep, XXXTentacion, Juice WRLD, Trippie Redd, GothBoiClique.

## Структура
```
corpus/                 10 новин (~3 200 слів), див. questions.md
schema.py               pydantic-схема: 6 типів сутностей, 13 типів зв'язків
llm.py                  клієнт Gemini (google-genai): генерація та ембединги
extract.py              п.2  -> output/facts.json
verify.py               п.2  -> output/verification.csv, verification_result.md
search.py               п.3  -> output/search_comparison.md
graph_rag.py            п.4  -> output/graph.png, graph.html, graphrag_vs_vectorrag.md
pipeline.py             все підряд
REPORT.md               звіт
```

## Запуск
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # вставити GEMINI_API_KEY (безкоштовно: aistudio.google.com/apikey)

python pipeline.py extract  # ~10 запитів до LLM
python pipeline.py verify   # -> output/verification.csv: заповнити correct (1/0) та error_type
python verify.py score      # частка коректних фактів
python pipeline.py search   # порівняння TF-IDF / підрядок / вектор
python pipeline.py graph    # граф + GraphRAG vs vector-RAG
python search.py хто помер від передозування   # довільний векторний запит
```
На безкоштовному рівні Gemini між запитами стоїть пауза 6.5 с (`LLM_MIN_INTERVAL`), увесь прогін ~5 хв.
Якщо модель недоступна — змініть `GEMINI_MODEL` у `.env` (напр. `gemini-2.0-flash`).
