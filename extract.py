# п.2 - екстракція фактів через LLM
import re

import config
from llm import generate_structured, save_json
from schema import Extraction

PROMPT = """Ти — система видобування знань для графа знань про емо-реп.
Прочитай новину і видобудь сутності та факти СТРОГО за схемою.

Правила:
1. Тільки те, що прямо сказано в тексті. Нічого не додавай із власних знань.
2. Артистів називай сценічним ім'ям: "Lil Peep", "XXXTentacion", "Juice WRLD", "Trippie Redd".
   Справжнє ім'я — лише в description. Інших людей — повним ім'ям: "Ліза Вомак", "Nick Mira".
   Релізи — без лапок і слів "альбом", "сингл": "Come Over When You're Sober, Pt. 2", "Lucid Dreams".
   Одна сутність = одна назва в усіх фактах.
3. Тип зв'язку — лише з дозволеного списку. subject — той, хто діє.
   RELEASED — головний виконавець або лейбл випустив реліз; FEATURED_ON — гість/співавтор треку;
   PRODUCED — продюсер; CHARTED_ON — реліз потрапив у чарт (позицію вкажи в evidence);
   PARTICIPATED_IN — людина в події (суд, тур, фестиваль, смерть як подія).
4. evidence — дослівне речення з документа.
5. year — рік із тексту або 0.
6. Видобудь усі суттєві факти (зазвичай 10–20).

Документ ({doc_id}):
\"\"\"
{text}
\"\"\"
"""

# синоніми, додавала після перегляду facts_raw.json
ALIASES = {
    "густав ар": "Lil Peep", "густав елайджа ар": "Lil Peep", "лил пип": "Lil Peep",
    "джасе онфрой": "XXXTentacion", "джасе двейн рікардо онфрой": "XXXTentacion",
    "джарад гіггінс": "Juice WRLD", "джарад ентоні гіггінс": "Juice WRLD",
    "майкл ламар вайт iv": "Trippie Redd",
    "адам макілві": "Wicca Phase Springs Eternal",
    "gbc": "GothBoiClique", "gothboi clique": "GothBoiClique",
    "columbia": "Columbia Records", "interscope": "Interscope Records",
    "grade a": "Grade A Productions",
    "hot 100": "Billboard Hot 100", "billboard 200 chart": "Billboard 200",
    "makonnen": "iLoveMakonnen",
}
STRIP_PREFIXES = ("альбом ", "мікстейп ", "сингл ", "пісня ", "трек ", "лейбл ", "фільм ", "колектив ", "чарт ")


def norm_key(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[«»\"'“”„]", "", s)
    s = re.sub(r"\s+", " ", s)
    for p in STRIP_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    return s


def canonicalize(facts, entities):
    canon = {}
    def register(name):
        k = norm_key(name)
        if k in ALIASES:
            return ALIASES[k]
        canon.setdefault(k, name.strip())
        return canon[k]

    for e in entities:
        e["name"] = register(e["name"])
    for f in facts:
        f["subject"] = register(f["subject"])
        f["object"] = register(f["object"])

    # 'Вомак' -> 'Ліза Вомак'
    persons = {e["name"] for e in entities if e["type"] == "Person"} | \
              {f["subject"] for f in facts if f["subject_type"] == "Person"} | \
              {f["object"] for f in facts if f["object_type"] == "Person"}
    full = [p for p in persons if " " in p]
    remap = {}
    for p in persons:
        if " " not in p:
            cands = [q for q in full if q.split()[-1] == p]
            if len(cands) == 1:
                remap[p] = cands[0]
    for f in facts:
        f["subject"] = remap.get(f["subject"], f["subject"])
        f["object"] = remap.get(f["object"], f["object"])
    for e in entities:
        e["name"] = remap.get(e["name"], e["name"])
    return facts, entities


CACHE_DIR = config.OUT_DIR / "extract_cache"


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    all_facts, all_entities = [], []
    for path in sorted(config.CORPUS_DIR.glob("*.txt")):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        cache = CACHE_DIR / f"{doc_id}.json"
        if cache.exists():  # вже оброблений - не витрачаємо запит
            res = Extraction.model_validate_json(cache.read_text(encoding="utf-8"))
            print(f"[extract] {doc_id} (кеш)", end=" ")
        else:
            print(f"[extract] {doc_id} ...", end=" ", flush=True)
            res = generate_structured(PROMPT.format(doc_id=doc_id, text=text), Extraction)
            cache.write_text(res.model_dump_json(indent=2), encoding="utf-8")
        for f in res.facts:
            d = f.model_dump(); d["doc_id"] = doc_id
            all_facts.append(d)
        for e in res.entities:
            d = e.model_dump(); d["doc_id"] = doc_id
            all_entities.append(d)
        print(f"{len(res.facts)} фактів, {len(res.entities)} сутностей")

    raw_n = len(all_facts)
    save_json(all_facts, config.OUT_DIR / "facts_raw.json")
    all_facts, all_entities = canonicalize(all_facts, all_entities)

    seen, facts = set(), []
    for f in all_facts:
        key = (f["subject"], f["predicate"], f["object"])
        if key in seen or f["subject"] == f["object"]:
            continue
        seen.add(key)
        f["id"] = f"F{len(facts) + 1:03d}"
        facts.append(f)

    ents = {}
    for e in all_entities:
        ents.setdefault(e["name"], e)

    save_json(facts, config.OUT_DIR / "facts.json")
    save_json(list(ents.values()), config.OUT_DIR / "entities.json")
    print(f"[extract] сирих фактів: {raw_n}, після нормалізації та дедуплікації: {len(facts)}; "
          f"сутностей: {len(ents)}")


if __name__ == "__main__":
    main()
