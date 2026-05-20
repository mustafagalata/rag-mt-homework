"""
RAG-MT mimarisinin görsel diyagramını üretir.
Çıktı: docs/rag_architecture.png

Kullanım:
    python docs/generate_diagram.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COL_OFFLINE = "#1f4e79"
COL_ONLINE = "#7e2f8e"
COL_DATA = "#fef3c7"
COL_MODEL = "#dbeafe"
COL_INDEX = "#dcfce7"
COL_LLM = "#fde2e4"
COL_TEXT = "#0f172a"
COL_ARROW = "#374151"
COL_LINK = "#1f4e79"


def box(ax, xy, w, h, text, face, edge="#1f2937", lw=1.4, fs=11):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=COL_TEXT)
    return (x, y, w, h)


def arrow(ax, p1, p2, label=None, lw=1.6, curve=0.0,
          label_offset=(0, 0.16), color=None):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color or COL_ARROW,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9, color="#475569", style="italic")


def bottom(b): return (b[0] + b[2] / 2, b[1])
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def left(b):   return (b[0], b[1] + b[3] / 2)
def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)


def main():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=180)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(7, 10.1, "RAG-based 5-shot Machine Translation",
            ha="center", fontsize=15, fontweight="bold", color="#0f172a")

    # Offline panel
    offline_bg = Rectangle((0.3, 6.3), 13.4, 3.2,
                           facecolor="#f8fafc", edgecolor=COL_OFFLINE,
                           linewidth=1.6, linestyle="--")
    ax.add_patch(offline_bg)
    ax.text(0.55, 9.15, "OFFLINE  ·  INDEXING",
            fontsize=11, fontweight="bold", color=COL_OFFLINE, ha="left")

    # Online panel
    online_bg = Rectangle((0.3, 0.3), 13.4, 5.5,
                          facecolor="#fdfaff", edgecolor=COL_ONLINE,
                          linewidth=1.6, linestyle="--")
    ax.add_patch(online_bg)
    ax.text(0.55, 5.45, "ONLINE  ·  TRANSLATION",
            fontsize=11, fontweight="bold", color=COL_ONLINE, ha="left")

    # ── OFFLINE ──────────────────────────────────────────────────────────
    corpus = box(ax, (0.7, 7.1), 2.8, 1.4,
                 "WMT16 corpus\n204,595 pairs", face=COL_DATA)

    embedder_off = box(ax, (4.0, 7.1), 3.0, 1.4,
                       "Embedder\n(e5-base)", face=COL_MODEL)

    embeddings = box(ax, (7.5, 7.1), 2.6, 1.4,
                     "Vectors", face=COL_DATA)

    faiss_idx = box(ax, (10.6, 7.1), 2.9, 1.4,
                    "FAISS index\n+ pair store", face=COL_INDEX)

    arrow(ax, right(corpus), left(embedder_off))
    arrow(ax, right(embedder_off), left(embeddings))
    arrow(ax, right(embeddings), left(faiss_idx))

    # ── ONLINE ───────────────────────────────────────────────────────────
    test_src = box(ax, (0.7, 3.9), 2.5, 1.2, "Test sentence", face=COL_DATA)

    embedder_on = box(ax, (3.8, 3.9), 2.6, 1.2, "Embedder", face=COL_MODEL)

    query_vec = box(ax, (7.0, 3.9), 1.7, 1.2, "Query Vector", face=COL_DATA, fs=12)

    retriever = box(ax, (9.3, 3.9), 4.2, 1.2,
                    "Retriever\ntop-5 pairs", face=COL_INDEX)

    arrow(ax, right(test_src), left(embedder_on))
    arrow(ax, right(embedder_on), left(query_vec))
    arrow(ax, right(query_vec), left(retriever))

    # cross-panel
    arrow(ax, bottom(faiss_idx), top(retriever),
          curve=0.0, color=COL_LINK, lw=1.6)

    # Prompt
    prompt_box = box(ax, (3.5, 2.2), 7.0, 1.0, "Prompt (5-shot)", face=COL_MODEL)
    # Retriever → prompt: prompt kutusunun üst kenarında sağ-merkez noktaya düz iniş
    prompt_top_right = (prompt_box[0] + prompt_box[2] * 0.85,
                        prompt_box[1] + prompt_box[3])
    arrow(ax, bottom(retriever), prompt_top_right, curve=0.0)

    # LLM
    llm = box(ax, (3.5, 0.7), 7.0, 1.0, "Qwen 2.5 7B", face=COL_LLM)
    arrow(ax, bottom(prompt_box), top(llm))

    # Output
    out = box(ax, (11.0, 0.8), 2.4, 0.9, "Translation", face=COL_DATA)
    arrow(ax, right(llm), left(out))

    out_path = Path(__file__).parent / "rag_architecture.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=180, facecolor="white")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
