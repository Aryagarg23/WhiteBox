"""
Toy stand-in for WhiteBox's Graph RAG pipeline, plus an explanatory diagram
of the real architecture.

The real system embeds a question, matches it against ~60k Neo4j nodes pulled
from a medical textbook, and walks the resulting subgraph to build an answer
(see server/Graph_RAG.py). Chart 1 fakes that on a ~25-node dict graph so the
core idea -- answer by walking a path you can point at, not by asking a model
to hallucinate one -- runs in a few seconds with no GPU and no Neo4j. The
walk itself is a real BFS on the toy graph below, not invented numbers.
Illustrative only: the graph is hand-written toy data, not the real KG.

Chart 2 is a concept sketch of the real pipeline as built at the hackathon:
boxes and arrows, no measurements, because none were collected.

Run: MPLCONFIGDIR=/home/arya/projects/hackathons/.mplcache \
     /home/arya/projects/hackathons/.venv/bin/python prototype/graph_qa_demo.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.style.use("/home/arya/projects/hackathons/.style/garg-paper.mplstyle")

from collections import deque

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

FEATURES = ["chest pain", "shortness of breath", "fatigue", "cough", "fever",
            "swelling in legs", "palpitations", "dizziness", "smoking history",
            "hypertension", "obesity", "diabetes"]
CONDITIONS = ["pneumonia", "heart failure", "myocardial infarction", "asthma",
              "anemia", "arrhythmia"]
TREATMENTS = ["antibiotics", "diuretics", "ACE inhibitors", "beta blockers",
              "bronchodilators", "iron supplements", "anticoagulants"]

EDGES = [
    ("chest pain", "myocardial infarction"), ("chest pain", "pneumonia"),
    ("shortness of breath", "heart failure"), ("shortness of breath", "asthma"),
    ("shortness of breath", "pneumonia"), ("fatigue", "anemia"),
    ("fatigue", "heart failure"), ("cough", "pneumonia"), ("cough", "asthma"),
    ("fever", "pneumonia"), ("swelling in legs", "heart failure"),
    ("palpitations", "arrhythmia"), ("dizziness", "arrhythmia"),
    ("dizziness", "anemia"), ("smoking history", "pneumonia"),
    ("smoking history", "myocardial infarction"), ("hypertension", "heart failure"),
    ("hypertension", "myocardial infarction"), ("obesity", "heart failure"),
    ("diabetes", "myocardial infarction"),
    ("pneumonia", "antibiotics"), ("heart failure", "diuretics"),
    ("heart failure", "ACE inhibitors"), ("heart failure", "beta blockers"),
    ("myocardial infarction", "beta blockers"),
    ("myocardial infarction", "anticoagulants"), ("asthma", "bronchodilators"),
    ("anemia", "iron supplements"), ("arrhythmia", "beta blockers"),
    ("arrhythmia", "anticoagulants"),
]


def build_adjacency():
    adjacency = {node: [] for node in FEATURES + CONDITIONS + TREATMENTS}
    for a, b in EDGES:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def bfs_path(adjacency, start, targets):
    """Shortest path from start to any node in targets. This is the whole
    'reasoning' step -- no model call, just a graph walk you can replay."""
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in targets:
            path = [node]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)
    return None


def node_positions():
    positions = {}
    for col, nodes in enumerate([FEATURES, CONDITIONS, TREATMENTS]):
        n = len(nodes)
        for i, node in enumerate(nodes):
            positions[node] = (col, (n - 1) / 2 - i)
    return positions


def plot_traced_path(adjacency, path, out_path):
    positions = node_positions()
    path_edges = set(zip(path, path[1:])) | set(zip(path[1:], path))

    fig, ax = plt.subplots(figsize=(10, 7))
    for a, b in EDGES:
        xa, ya = positions[a]
        xb, yb = positions[b]
        on_path = (a, b) in path_edges
        ax.plot([xa, xb], [ya, yb],
                color="#3b42db" if on_path else "#282215",
                linewidth=2.5 if on_path else 0.5,
                alpha=1.0 if on_path else 0.25,
                zorder=2 if on_path else 1)

    colors = {**{n: "#3b42db" for n in FEATURES},
              **{n: "#c2491d" for n in CONDITIONS},
              **{n: "#6f2f96" for n in TREATMENTS}}
    for node, (x, y) in positions.items():
        on_path = node in path
        ax.scatter(x, y, s=140 if on_path else 45,
                   color=colors[node], edgecolor="#282215" if on_path else "none",
                   linewidth=1.5, zorder=3)
        ax.annotate(node, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=7 if on_path else 6,
                    fontweight="bold" if on_path else "normal")

    ax.set_title(f"How does '{path[0]}' trace to '{path[-1]}'? "
                 f"{len(path) - 1}-hop path through a toy graph")
    ax.text(0.0, 1.06, "toy graph, real walk (BFS on the 25-node graph below,"
            " not the real ~60k-node KG)", transform=ax.transAxes,
            fontsize=8, style="italic", color="#57503f")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["presenting features", "conditions", "treatments"])
    ax.set_yticks([])
    fig.savefig(out_path)
    plt.close(fig)


def _box(ax, xy, w, h, text, color, fontsize=9, fontweight="bold"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0", linewidth=1.4,
        edgecolor="#282215", facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="#282215",
            wrap=True, zorder=3)


def _arrow(ax, start, end, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        linewidth=1.4, color="#282215", zorder=1))


def plot_pipeline(out_path):
    """Concept sketch of the real WhiteBox architecture as built at the
    hackathon. No measurements -- boxes and arrows only, plus the two real
    build-time figures (~60k nodes / ~130k relationships) called out on the
    Neo4j box, since those came from the actual graph load, not an estimate."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    cream = "#eae4d6"
    build_color = "#c6b99f"
    query_color = "#dcd3bb"
    accent_blue = "#8f96f0"
    accent_purple = "#c9aee0"

    ax.text(0, 6.7, "How WhiteBox turns a textbook into a traceable answer",
            fontsize=13, fontweight="medium", ha="left")
    ax.text(0, 6.35, "architecture as built at the hackathon -- no runtime "
            "measurements, this is a map of the pipeline, not a benchmark",
            fontsize=8.5, style="italic", color="#57503f")

    # --- build-time row (top): textbook -> chunk/classify -> graph ---
    ax.text(0, 5.65, "build time (once)", fontsize=8.5, fontweight="bold",
            color="#57503f")
    _box(ax, (0, 4.7), 2.3, 0.8, "Davidson's Principles\n& Practice of Medicine",
         cream)
    _box(ax, (3.15, 4.7), 2.5, 0.8, "chunk + classify\n(fine-tuned LLaMA 3.1 8B)",
         build_color)
    _box(ax, (6.45, 4.55), 2.9, 1.1,
         "Neo4j graph\n~60k nodes / ~130k relationships\n(real figures from the build)",
         accent_blue, fontsize=8.5)
    _arrow(ax, (2.3, 5.1), (3.15, 5.1))
    _arrow(ax, (5.65, 5.1), (6.45, 5.1))

    # --- query-time row (bottom): question -> MiniLM -> Cypher -> Phi-3 -> answer ---
    ax.text(0, 3.55, "query time (per question)", fontsize=8.5,
            fontweight="bold", color="#57503f")
    _box(ax, (0, 2.6), 1.9, 0.8, "clinician's\nquestion", cream)
    _box(ax, (2.55, 2.6), 2.0, 0.8, "MiniLM\nembed + score terms", build_color)
    _box(ax, (5.2, 2.6), 2.1, 0.8, "Cypher retrieval\ntop-3 candidates", query_color)
    _box(ax, (8.0, 2.6), 1.9, 0.8, "Phi-3-mini\nanalysis / generation", accent_purple)
    _arrow(ax, (1.9, 3.0), (2.55, 3.0))
    _arrow(ax, (4.55, 3.0), (5.2, 3.0))
    _arrow(ax, (7.3, 3.0), (8.0, 3.0))

    # graph feeds the query-time retrieval step
    _arrow(ax, (7.9, 4.55), (6.25, 3.4), style="-|>")

    # --- answer row (bottom): answer + path -> clinician ---
    _box(ax, (3.85, 1.0), 4.3, 0.9,
         "answer, returned WITH the graph path\nit was built from",
         "#f2c9b8", fontsize=9)
    _arrow(ax, (8.95, 2.6), (7.2, 1.9))
    _arrow(ax, (5.2, 1.0), (2.0, 0.35))
    ax.text(0.0, 0.05, "clinician sees answer + path, can audit either",
            fontsize=8.5, color="#57503f")

    fig.savefig(out_path)
    plt.close(fig)


def main():
    adjacency = build_adjacency()
    start = "shortness of breath"
    path = bfs_path(adjacency, start, set(TREATMENTS))

    print(f"Question: what treats '{start}'?")
    print("Traced path:", " -> ".join(path))

    traced_path_out = os.path.join(FIGDIR, "traced_path.png")
    pipeline_out = os.path.join(FIGDIR, "pipeline.png")
    plot_traced_path(adjacency, path, traced_path_out)
    plot_pipeline(pipeline_out)
    print(f"Saved {traced_path_out} and {pipeline_out}")


if __name__ == "__main__":
    main()
