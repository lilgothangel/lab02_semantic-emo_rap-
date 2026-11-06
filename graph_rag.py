import math
from collections import Counter
from pathlib import Path

import community as community_louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel, Field

import config
from llm import generate_structured, generate_text, load_json, save_json

MIN_COMMUNITY = 3


def build_graph(facts):
    G = nx.DiGraph()
    for f in facts:
        for n, t in ((f["subject"], f["subject_type"]), (f["object"], f["object_type"])):
            if n not in G:
                G.add_node(n, type=t, docs=set())
            G.nodes[n]["docs"].add(f["doc_id"])
        G.add_edge(f["subject"], f["object"], predicate=f["predicate"],
                   year=f["year"], evidence=f["evidence"], doc_id=f["doc_id"], fact_id=f["id"])
    return G


def detect_communities(G):
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes)
    for u, v in G.edges:
        w = UG[u][v]["weight"] + 1 if UG.has_edge(u, v) else 1
        UG.add_edge(u, v, weight=w)
    raw = community_louvain.best_partition(UG, random_state=42)
    mod = community_louvain.modularity(raw, UG)
    # дрібні спільноти (1-2 вузли) приєднуємо до спільноти найзв'язанішого сусіда
    sizes = Counter(raw.values())
    part = dict(raw)
    for n in sorted(UG, key=lambda x: sizes[raw[x]]):
        if sizes[raw[n]] < MIN_COMMUNITY and UG.degree(n):
            nb = max(UG[n], key=lambda m: (sizes[part[m]], UG.degree(m)))
            part[n] = part[nb]
    # перенумерація за розміром: 0 - найбільша; обривки з 1-2 вузлів -> одна група «інше»
    cnt = Counter(part.values())
    order = [c for c, k in cnt.most_common() if k >= MIN_COMMUNITY]
    misc = len(order)
    part = {n: (order.index(c) if c in order else misc) for n, c in part.items()}
    G.graph["misc"] = misc if misc in part.values() else None
    return part, mod


SHAPES = {"Person": "o", "Organization": "s", "Release": "D", "Place": "^", "Event": "p", "Chart": "h"}
TYPE_UA = {"Person": "Людина", "Organization": "Організація / лейбл", "Release": "Реліз",
           "Place": "Місце", "Event": "Подія", "Chart": "Чарт"}


def radial_layout(sub, hub):
    # хаб у центрі, сусіди по колу, їхні сусіди - на зовнішньому колі біля «батька»
    parent, level, order = {hub: None}, {hub: 0}, [hub]
    for u in order:
        for v in sorted(sub[u], key=lambda x: -sub.degree(x)):
            if v not in level:
                level[v], parent[v] = level[u] + 1, u
                order.append(v)
    kids = {n: [k for k in order if parent.get(k) == n] for n in order}
    weight = {}
    for n in reversed(order):  # скільки місця по колу потрібно гілці
        weight[n] = max(1.0, sum(weight[k] for k in kids[n]))
    ring1 = kids[hub]
    total = sum(weight[n] for n in ring1) or 1
    r1 = 1.0 + 0.05 * len(ring1)
    pos, span = {hub: (0.0, 0.0)}, {}
    start = math.pi / 2
    for n in ring1:
        w = 2 * math.pi * weight[n] / total
        span[n] = (start, start + w)
        start += w
    for n in order[1:]:
        lo, hi = span[n]
        a = (lo + hi) / 2
        r = r1 + 0.9 * (level[n] - 1)
        pos[n] = (r * math.cos(a), r * math.sin(a))
        cur = lo
        for k in kids[n]:
            w = (hi - lo) * weight[k] / weight[n]
            span[k] = (cur, cur + w)
            cur += w
    return pos


def community_layout(G, part):
    # кожна спільнота - окремий кластер, кластери по сітці
    comms = sorted(set(part.values()))
    misc = G.graph.get("misc")
    local, radius = {}, {}
    for c in comms:
        nodes = [n for n in G if part[n] == c]
        sub = G.subgraph(nodes).to_undirected()
        p = {}
        if c == misc:
            # «інше»: дрібні компоненти рядами
            comps = sorted(nx.connected_components(sub), key=len, reverse=True)
            cols = math.ceil(math.sqrt(len(comps)))
            for i, comp in enumerate(comps):
                h = max(comp, key=sub.degree)
                rp = radial_layout(sub.subgraph(comp), h)
                ox, oy = (i % cols) * 2.0, -(i // cols) * 1.4
                p.update({n: (ox + x * 0.6, oy + y * 0.6) for n, (x, y) in rp.items()})
            cx = sum(x for x, _ in p.values()) / len(p); cy = sum(y for _, y in p.values()) / len(p)
            p = {n: (x - cx, y - cy) for n, (x, y) in p.items()}
        else:
            # спільнота може бути незв'язною всередині - кожну частину від свого хаба
            for comp in sorted(nx.connected_components(sub), key=len, reverse=True):
                h = max(comp, key=lambda n: G.degree(n))
                rp = radial_layout(sub.subgraph(comp), h)
                if p:
                    off = max(math.hypot(x, y) for x, y in p.values()) + 0.8
                    rp = {n: (x + off, y) for n, (x, y) in rp.items()}
                p.update(rp)
        local[c] = p
        radius[c] = max([math.hypot(x, y) for x, y in p.values()] + [0.8])
    cols = math.ceil(math.sqrt(len(comms)))
    cell_w = 2 * max(radius.values()) + 1.0
    pos, y0 = {}, 0.0
    for row in range(0, len(comms), cols):
        rc = comms[row:row + cols]
        h = max(radius[c] for c in rc)
        y0 -= h
        for j, c in enumerate(rc):
            for n, (x, y) in local[c].items():
                pos[n] = (j * cell_w + x, y0 + y)
        y0 -= h + 1.0
    return pos


def draw(G, part, path_png, path_html):
    pos = community_layout(G, part)
    cmap = plt.get_cmap("tab10" if len(set(part.values())) <= 10 else "tab20")
    color = {n: cmap(part[n] % cmap.N) for n in G}
    deg = dict(G.degree())
    sizes = Counter(part.values())
    hubs = {max((n for n in G if part[n] == c), key=lambda n: deg[n])
            for c in sizes if sizes[c] >= MIN_COMMUNITY and c != G.graph.get("misc")}

    fig, ax = plt.subplots(figsize=(26, 20))
    intra = [(u, v) for u, v in G.edges if part[u] == part[v]]
    inter = [(u, v) for u, v in G.edges if part[u] != part[v]]
    nx.draw_networkx_edges(G, pos, edgelist=inter, edge_color="#bbbbbb", alpha=0.45, width=0.8,
                           arrows=True, arrowsize=7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=intra, edge_color="#888888", alpha=0.6, width=0.9,
                           arrows=True, arrowsize=8, ax=ax)
    for t, shape in SHAPES.items():
        nodes = [n for n in G if G.nodes[n]["type"] == t]
        if not nodes:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shape, ax=ax,
                               node_color=[color[n] for n in nodes], edgecolors="white", linewidths=1,
                               node_size=[2600 if n in hubs else 140 + 40 * deg[n] for n in nodes])
    for n, (x, y) in pos.items():
        if n in hubs:
            ax.text(x, y, n, fontsize=13, fontweight="bold", ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
        else:
            ax.text(x, y - 0.12, n, fontsize=8.5, ha="center", va="top")
    handles = [plt.Line2D([], [], marker=m, color="gray", linestyle="", markersize=12, label=TYPE_UA[t])
               for t, m in SHAPES.items() if any(G.nodes[n]["type"] == t for n in G)]
    ax.legend(handles=handles, loc="lower left", fontsize=13, frameon=True)
    ax.set_title("Граф знань: емо-реп (колір - спільнота Louvain, форма - тип сутності, великий вузол - хаб)",
                 fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path_png, dpi=130)
    plt.close(fig)

    try:
        from pyvis.network import Network
        net = Network(height="900px", width="100%", directed=True, cdn_resources="remote",
                      bgcolor="#ffffff")
        vis_shape = {"Person": "dot", "Organization": "square", "Release": "diamond",
                     "Place": "triangle", "Event": "star", "Chart": "hexagon"}
        for n in G:
            net.add_node(n, label=n, color=matplotlib.colors.to_hex(color[n]),
                         shape=vis_shape.get(G.nodes[n]["type"], "dot"),
                         size=35 if n in hubs else 8 + 3 * deg[n],
                         title=f"{TYPE_UA.get(G.nodes[n]['type'])} | спільнота {part[n]}")
        for u, v, d in G.edges(data=True):
            net.add_edge(u, v, title=f"{d['predicate']}: {d['evidence']}", label=d["predicate"],
                         font={"size": 0}, arrows="to")
        net.force_atlas_2based(gravity=-60, spring_length=120)
        # pyvis на Windows пише файл не в utf-8 -> кирилиця ламає html, тому пишемо самі
        html = net.generate_html(notebook=False)
        Path(path_html).write_text(html, encoding="utf-8")
    except Exception as e:
        print("[graph] pyvis пропущено:", e)


class CommunityReport(BaseModel):
    title: str = Field(description="Коротка назва спільноти (3–7 слів)")
    summary: str = Field(description="Резюме 4–7 речень: хто/що входить, ключові зв'язки, роль у темі")
    key_entities: list[str]


class PartialAnswer(BaseModel):
    relevance: int = Field(description="Наскільки резюме корисне для запитання, 0–100")
    points: list[str] = Field(description="Ключові тези з цього резюме для відповіді (порожньо, якщо нерелевантне)")


SUMMARY_PROMPT = """Нижче — фрагмент графа знань (спільнота) про емо-реп: сутності та зв'язки.
Склади звіт про цю спільноту українською мовою. Спирайся ЛИШЕ на наведені зв'язки.

Сутності:
{entities}

Зв'язки (суб'єкт -[тип]-> об'єкт, рік):
{edges}
"""

MAP_PROMPT = """Запитання: {question}

Звіт про одну тематичну спільноту графа знань:
Назва: {title}
{summary}

Оціни релевантність звіту для запитання (0–100) та випиши тези з нього, що допомагають відповісти.
Використовуй тільки інформацію зі звіту."""

REDUCE_PROMPT = """Запитання: {question}

Нижче — тези, зібрані з резюме різних тематичних спільнот графа знань (відсортовані за релевантністю):
{points}

Напиши повну, структуровану відповідь українською (200–350 слів). Об'єднай тези, покажи зв'язки
між людьми та організаціями з різних спільнот, зроби узагальнення. Не додавай фактів, яких немає в тезах."""

VECTOR_RAG_PROMPT = """Дай відповідь на запитання українською (200–350 слів), спираючись ЛИШЕ на фрагменти нижче.

Запитання: {question}

Фрагменти:
{context}"""


def summarize_communities(G, part):
    reports = []
    for c in sorted(set(part.values())):
        nodes = [n for n in G if part[n] == c]
        if c == G.graph.get("misc") or len(nodes) < MIN_COMMUNITY:
            continue
        sub = G.subgraph(nodes)
        # + ребра назовні спільноти
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if u in nodes or v in nodes]
        ent_txt = "\n".join(f"- {n} ({G.nodes[n]['type']})" for n in nodes)
        edge_txt = "\n".join(f"- {u} -[{d['predicate']}]-> {v}" + (f", {d['year']}" if d["year"] else "")
                             for u, v, d in edges)
        print(f"[graphrag] резюме спільноти {c} ({len(nodes)} вузлів, {sub.number_of_edges()} внутр. ребер)")
        r = generate_structured(SUMMARY_PROMPT.format(entities=ent_txt, edges=edge_txt), CommunityReport)
        reports.append({"community": c, "size": len(nodes), "nodes": nodes, **r.model_dump()})
    save_json(reports, config.OUT_DIR / "community_reports.json")
    return reports


def graphrag_answer(question, reports):
    partials = []
    for r in reports:  # map
        p = generate_structured(MAP_PROMPT.format(question=question, title=r["title"], summary=r["summary"]),
                                PartialAnswer)
        print(f"[graphrag] map: «{r['title']}» релевантність {p.relevance}")
        if p.relevance > 0 and p.points:
            partials.append((p.relevance, r["title"], p.points))
    partials.sort(key=lambda x: -x[0])
    pts = "\n".join(f"[{t}, релевантність {rel}]\n" + "\n".join(f"- {x}" for x in ps)
                    for rel, t, ps in partials)
    answer = generate_text(REDUCE_PROMPT.format(question=question, points=pts))  # reduce
    return answer, partials


def vector_rag_answer(question):
    from search import VectorStore
    hits = VectorStore().search(question, k=config.TOP_N)
    ctx = "\n\n".join(f"[{c['id']}] {c['text']}" for c, _ in hits)
    return generate_text(VECTOR_RAG_PROMPT.format(question=question, context=ctx)), hits


def main(question=config.GLOBAL_QUESTION, draw_only=False):
    facts = load_json(config.OUT_DIR / "facts.json")
    G = build_graph(facts)
    part, mod = detect_communities(G)
    nx.set_node_attributes(G, part, "community")
    print(f"[graph] вузлів {G.number_of_nodes()}, ребер {G.number_of_edges()}, "
          f"спільнот {len(set(part.values()) - {G.graph.get('misc')})}, модулярність {mod:.3f}")
    draw(G, part, config.OUT_DIR / "graph.png", config.OUT_DIR / "graph.html")
    print("[graph] збережено output/graph.png, output/graph.html")
    if draw_only:
        return

    reports = summarize_communities(G, part)
    g_ans, partials = graphrag_answer(question, reports)
    v_ans, hits = vector_rag_answer(question)

    top_deg = sorted(G.degree(), key=lambda x: -x[1])[:10]
    md = [f"# GraphRAG vs vector-RAG\n",
          f"**Запитання:** {question}\n",
          f"Граф: {G.number_of_nodes()} вузлів, {G.number_of_edges()} ребер, "
          f"{len(reports)} спільнот (Louvain, модулярність {mod:.3f}).\n",
          "Найбільш зв'язані вузли: " + ", ".join(f"{n} ({d})" for n, d in top_deg) + "\n",
          "## Спільноти\n"]
    md += [f"- **{r['community']}. {r['title']}** ({r['size']} вузлів): {r['summary']}" for r in reports]
    md += ["\n## Відповідь GraphRAG\n", g_ans,
           "\n## Відповідь vector-RAG (top-%d чанків)\n" % config.TOP_N, v_ans,
           "\nВикористані чанки: " + ", ".join(f"`{c['id']}`" for c, _ in hits),
           "\nДокументи, що покриті vector-RAG: " + ", ".join(sorted({c['doc_id'] for c, _ in hits}))]
    (config.OUT_DIR / "graphrag_vs_vectorrag.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    import sys
    main(draw_only="draw" in sys.argv)  # python graph_rag.py draw - тільки картинка, без LLM