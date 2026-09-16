from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "manuscript"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
RESULT_ROOT = Path(r"C:\Users\杨昊\Desktop\实验数据")
OWL_ROOT = RESULT_ROOT / "owl" / "owl"

DOCX_PATH = OUT_DIR / "GS-OWL_计算机工程与应用_论文初稿.docx"

NAVY = "17365D"
BLUE = "2F6EA5"
TEAL = "2B7A78"
ORANGE = "C9652B"
INK = "22252A"
MUTED = "5D6670"
LIGHT_BLUE = "EAF2F8"
LIGHT_TEAL = "E9F4F2"
LIGHT_ORANGE = "FBEFE7"
LIGHT_GRAY = "F3F5F7"
MID_GRAY = "D9DEE3"
WHITE = "FFFFFF"

CN_BODY = "宋体"
CN_HEAD = "黑体"
EN_FONT = "Times New Roman"
MATH_FONT = "Cambria Math"


PPL_RESULTS = {
    "LLaMA-1-7B": {
        "Magnitude": [17.2625, 561.7321, 48835.1914],
        "Wanda": [7.9436, 10.7007, 80.7771],
        "SparseGPT": [7.2099, 10.2422, 26.3452],
        "RIA": [7.1225, 10.5735, 92.5371],
        "Pruner-Zero": [6.8940, 9.9361, 70.0180],
        "GBLM": [7.2020, 10.4819, 79.1143],
        "GS-OWL": [6.9126, 8.8426, 20.8710],
    },
    "LLaMA-1-13B": {
        "Magnitude": [20.1370, 224.6088, 84531.1484],
        "Wanda": [6.5734, 8.7563, 53.1184],
        "SparseGPT": [6.2108, 8.4268, 19.5350],
        "RIA": [6.0763, 8.5370, 72.9075],
        "Pruner-Zero": [5.9441, 7.6828, 35.1296],
        "GBLM": [6.1407, 8.6931, 54.1907],
        "GS-OWL": [5.8731, 7.0694, 11.9627],
    },
    "LLaMA-2-7B": {
        "Magnitude": [14.8958, 3677.5183, 52457.0586],
        "Wanda": [7.7839, 10.0450, 70.9594],
        "SparseGPT": [6.5154, 9.5809, 24.2518],
        "RIA": [6.8067, 10.3874, 65.2370],
        "Pruner-Zero": [6.2617, 9.6862, 100.8275],
        "GBLM": [6.4295, 9.8600, 63.8275],
        "GS-OWL": [6.1975, 8.1455, 25.5248],
    },
    "LLaMA-2-13B": {
        "Magnitude": [None, None, None],
        "Wanda": [6.2838, 7.9192, 45.2404],
        "SparseGPT": [5.6272, 7.8273, 18.0321],
        "RIA": [5.8275, 7.8384, 52.4052],
        "Pruner-Zero": [5.3632, 6.8989, 27.7549],
        "GBLM": [5.5606, 7.8498, 41.6285],
        "GS-OWL": [5.3435, 6.5466, 12.2744],
    },
}

DENSE_PPL = {
    "LLaMA-1-7B": 5.68,
    "LLaMA-1-13B": 5.09,
    "LLaMA-2-7B": 5.47,
    "LLaMA-2-13B": 4.88,
}

ZERO_SHOT = {
    "LLaMA-1-7B": {
        "Magnitude": [46.9143, 36.0857, 32.2571],
        "Wanda": [52.5214, 50.3143, 37.4971],
        "SparseGPT": [55.0929, 51.0014, 42.4000],
        "RIA": [55.9471, 49.3586, 36.5100],
        "Pruner-Zero": [54.5757, 49.0800, 38.2171],
        "GS-OWL": [54.3070, 49.2300, 43.3260],
    },
    "LLaMA-1-13B": {
        "Magnitude": [47.4971, 41.0414, 34.9857],
        "Wanda": [58.3343, 53.4871, 39.0486],
        "SparseGPT": [58.6829, 53.4614, 45.3300],
        "RIA": [58.8700, 54.1357, 39.9171],
        "Pruner-Zero": [58.2971, 52.1714, 41.1029],
        "GS-OWL": [58.7590, 53.6840, 46.7810],
    },
    "LLaMA-2-7B": {
        "Magnitude": [51.1471, 39.5800, 33.3629],
        "Wanda": [54.6300, 50.2500, 35.3200],
        "SparseGPT": [56.0486, 51.9071, 41.7743],
        "RIA": [55.4543, 50.2200, 36.6943],
        "Pruner-Zero": [54.3686, 48.5029, 34.2457],
        "GS-OWL": [55.2400, 49.7130, 43.5110],
    },
    "LLaMA-2-13B": {
        "Magnitude": [None, None, None],
        "Wanda": [58.8800, 55.5186, 37.3514],
        "SparseGPT": [60.7814, 56.1529, 45.7657],
        "RIA": [59.4614, 56.1529, 41.1700],
        "Pruner-Zero": [59.2543, 54.7457, 41.0014],
        "GS-OWL": [59.4830, 55.5630, 47.2430],
    },
}

FINAL_GENERALIZATION = {
    "LLaMA-1-30B": [4.9902, 5.9740, 9.0048],
    "Mistral-7B": [5.6866, 7.9216, 25.3744],
    "OPT-1.3B": [22.0914, 40.6130, 350.7006],
}

ABLATION = [
    ("LLaMA-1-7B", 0.5, 7.3649, 7.2944, 7.2900, 6.9549, 6.9131, 6.9126),
    ("LLaMA-1-7B", 0.6, 10.9298, 9.5862, 9.5629, 9.9098, 8.8608, 8.8426),
    ("LLaMA-1-7B", 0.7, 89.2707, 22.3094, 21.9540, 68.5487, 21.1797, 20.8710),
    ("LLaMA-2-7B", 0.5, 6.4621, 6.3819, 6.3764, 6.2605, 6.2041, 6.1975),
    ("LLaMA-2-7B", 0.6, 10.0450, 8.4773, 8.4406, 9.6475, 8.1845, 8.1454),
    ("LLaMA-2-7B", 0.7, 70.9594, 20.2903, 19.4580, 112.8848, 26.5785, 25.5227),
]

REFERENCES = [
    "TOUVRON H, LAVRIL T, IZACARD G, et al. LLaMA: Open and efficient foundation language models[EB/OL]. arXiv:2302.13971, 2023.",
    "TOUVRON H, MARTIN L, STONE K, et al. Llama 2: Open foundation and fine-tuned chat models[EB/OL]. arXiv:2307.09288, 2023.",
    "VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017: 5998-6008.",
    "LECUN Y, DENKER J S, SOLLA S A. Optimal brain damage[C]//Advances in Neural Information Processing Systems. 1989: 598-605.",
    "HASSIBI B, STORK D G. Second order derivatives for network pruning: Optimal brain surgeon[C]//Advances in Neural Information Processing Systems. 1993: 164-171.",
    "HAN S, POOL J, TRAN J, et al. Learning both weights and connections for efficient neural networks[C]//Advances in Neural Information Processing Systems. 2015: 1135-1143.",
    "HAN S, MAO H, DALLY W J. Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding[C]//International Conference on Learning Representations. 2016.",
    "FRANTAR E, ALISTARH D. SparseGPT: Massive language models can be accurately pruned in one-shot[C]//Proceedings of the 40th International Conference on Machine Learning. 2023: 10323-10337.",
    "SUN M, LIU Z, BAIR A, et al. A simple and effective pruning approach for large language models[C]//International Conference on Learning Representations. 2024.",
    "YIN L, WU Y, ZHANG Z, et al. Outlier weighed layerwise sparsity: A missing ingredient for pruning large language models[C]//Proceedings of the 41st International Conference on Machine Learning. 2024.",
    "ZHANG Y, BAI H, LIN H, et al. Plug-and-play: An efficient post-training pruning method for large language models[C]//International Conference on Learning Representations. 2024.",
    "DONG P, LI L, WEI Z, et al. Pruner-Zero: Evolving symbolic pruning metric from scratch for large language models[C]//Proceedings of the 41st International Conference on Machine Learning. 2024.",
    "MA X, FANG G, WANG X. LLM-Pruner: On the structural pruning of large language models[C]//Advances in Neural Information Processing Systems. 2023, 36: 21702-21720.",
    "ASHKBOOS S, CROCI M L, NASIR A A, et al. SliceGPT: Compress large language models by deleting rows and columns[C]//International Conference on Learning Representations. 2024.",
    "FRANTAR E, ASHKBOS S, HOEFLER T, et al. GPTQ: Accurate post-training quantization for generative pre-trained transformers[C]//International Conference on Learning Representations. 2023.",
    "LIN J, TANG J, TANG H, et al. AWQ: Activation-aware weight quantization for LLM compression and acceleration[C]//Proceedings of Machine Learning and Systems. 2024, 6: 87-100.",
    "DETTMERS T, LEWIS M, BELKADA Y, et al. LLM.int8(): 8-bit matrix multiplication for transformers at scale[C]//Advances in Neural Information Processing Systems. 2022, 35: 30318-30332.",
    "MERITY S, XIONG C, BRADBURY J, et al. Pointer sentinel mixture models[C]//International Conference on Learning Representations. 2017.",
    "CLARK C, LEE K, CHANG M W, et al. BoolQ: Exploring the surprising difficulty of natural yes/no questions[C]//Proceedings of NAACL-HLT. 2019: 2924-2936.",
    "ZELLERS R, HOLTZMAN A, BISK Y, et al. HellaSwag: Can a machine really finish your sentence?[C]//Proceedings of ACL. 2019: 4791-4800.",
    "SAKAGUCHI K, LE BRAS R, BHAGAVATULA C, et al. WinoGrande: An adversarial Winograd schema challenge at scale[C]//Proceedings of AAAI. 2020, 34(5): 8732-8740.",
    "MIHAYLOV T, CLARK P, KORDJAMSHIDI P, et al. Can a suit of armor conduct electricity? A new dataset for open book question answering[C]//Proceedings of EMNLP. 2018: 2381-2391.",
    "CHEN Y, et al. DLP: Dynamic layer-wise pruning for large language models[C]//Proceedings of the 42nd International Conference on Machine Learning. 2025.",
    "CHEN Y, et al. LLM-Streamline: Layer pruning guided by similarity and uncertainty[C]//International Conference on Learning Representations. 2025.",
    "GU J, et al. DenoiseRotator: Rotated denoising for robust large language model pruning[C]//Advances in Neural Information Processing Systems. 2025.",
    "LI Y, et al. Týr-the-Pruner: Adaptive structural pruning for large language models[C]//Advances in Neural Information Processing Systems. 2025.",
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def rounded_box(ax, xy, width, height, title, lines, edge, fill):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=f"#{edge}",
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height - 0.075, title, ha="center", va="center", fontsize=11, weight="bold", color=f"#{INK}")
    for idx, line in enumerate(lines):
        ax.text(
            x + width / 2,
            y + height - 0.19 - idx * 0.075,
            line,
            ha="center",
            va="center",
            fontsize=9,
            color=f"#{MUTED if idx else edge}",
        )


def make_overview_figure() -> Path:
    configure_matplotlib()
    path = FIG_DIR / "gs_owl_method_overview.png"
    fig, ax = plt.subplots(figsize=(12.2, 4.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(
        ax,
        (0.025, 0.18),
        0.205,
        0.64,
        "输入与校准",
        ["预训练语言模型", "C4 校准样本", "权重 W / 激活 X / 梯度 G"],
        BLUE,
        "#EAF2F8",
    )
    rounded_box(
        ax,
        (0.285, 0.18),
        0.27,
        0.64,
        "OWL-V9：层间稀疏率分配",
        ["异常证据  O=|W| sqrt(X)", "计数比例 c + 超阈严重度 v", "锚定融合 + 核心层保护", "输出每层稀疏率 s_l"],
        TEAL,
        "#E9F4F2",
    )
    rounded_box(
        ax,
        (0.61, 0.18),
        0.225,
        0.64,
        "wsqrtg：层内权重排序",
        ["显著性  M=|W| sqrt(|G|)", "按 s_l 逐层、逐行选阈值", "保留梯度敏感权重"],
        ORANGE,
        "#FBEFE7",
    )
    rounded_box(
        ax,
        (0.885, 0.18),
        0.09,
        0.64,
        "输出",
        ["非均匀", "稀疏模型", "无需微调"],
        NAVY,
        "#EDF1F6",
    )

    for x0, x1, color in [(0.23, 0.285, BLUE), (0.555, 0.61, TEAL), (0.835, 0.885, ORANGE)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.006, 0.5),
                (x1 - 0.006, 0.5),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color=f"#{color}",
            )
        )

    # Small schematic matrices and layer bars provide technical texture without crowding labels.
    for r in range(3):
        for c in range(5):
            ax.add_patch(
                Rectangle(
                    (0.065 + c * 0.025, 0.29 + r * 0.035),
                    0.019,
                    0.025,
                    facecolor=["#BFD5E6", "#7EADD0", "#3D7FB0"][(r + c) % 3],
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
    layer_values = [0.53, 0.49, 0.50, 0.55, 0.60, 0.64, 0.66, 0.62]
    for i, value in enumerate(layer_values):
        ax.add_patch(
            Rectangle(
                (0.32 + i * 0.026, 0.215),
                0.018,
                0.18 * value,
                facecolor=f"#{TEAL}",
                alpha=0.85,
                edgecolor=f"#{WHITE}",
                linewidth=0.4,
            )
        )
    for r in range(3):
        for c in range(5):
            keep = (r + c) % 3 != 1
            ax.add_patch(
                Rectangle(
                    (0.65 + c * 0.026, 0.29 + r * 0.035),
                    0.020,
                    0.025,
                    facecolor=f"#{ORANGE if keep else WHITE}",
                    edgecolor="#D2B6A7",
                    linewidth=0.5,
                )
            )
    fig.tight_layout(pad=0.5)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_layer_figure() -> Path:
    configure_matplotlib()
    source = OWL_ROOT / "analysis" / "paper_visualizations" / "figure_layer_allocation_data.csv"
    data = pd.read_csv(source)
    path = FIG_DIR / "layer_allocation_compact.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=220, sharey=True)
    colors = {"uniform_sparsity": "#6B7280", "original_owl_sparsity": f"#{BLUE}", "owl_v9_sparsity": f"#{ORANGE}"}
    labels = {"uniform_sparsity": "均匀剪枝", "original_owl_sparsity": "原始 OWL", "owl_v9_sparsity": "OWL-V9"}
    for ax, key, title in zip(axes, ["llama1_7b", "llama2_7b"], ["LLaMA-1-7B", "LLaMA-2-7B"]):
        sub = data[data["model_key"] == key]
        for col in ["uniform_sparsity", "original_owl_sparsity", "owl_v9_sparsity"]:
            ax.plot(
                sub["layer"],
                100 * sub[col],
                color=colors[col],
                linewidth=2.0,
                linestyle="--" if col == "uniform_sparsity" else "-",
                label=labels[col],
            )
        ax.axvspan(4, 10, color="#DCEFEB", alpha=0.6, zorder=0)
        ax.axvspan(16, 30, color="#F8E9DF", alpha=0.45, zorder=0)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Transformer 层")
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("层稀疏率 / %")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_main_result_figure() -> Path:
    configure_matplotlib()
    path = FIG_DIR / "main_ppl_s06.png"
    models = list(PPL_RESULTS.keys())
    methods = ["Wanda", "SparseGPT", "Pruner-Zero", "GS-OWL"]
    x = np.arange(len(models))
    width = 0.19
    colors = ["#A9B0B8", f"#{BLUE}", f"#{TEAL}", f"#{ORANGE}"]
    fig, ax = plt.subplots(figsize=(9.3, 3.9), dpi=220)
    for idx, (method, color) in enumerate(zip(methods, colors)):
        vals = [PPL_RESULTS[m][method][1] for m in models]
        bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=method, color=color, edgecolor="white", linewidth=0.6)
        if method == "GS-OWL":
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, weight="bold")
    ax.set_xticks(x, ["L1-7B", "L1-13B", "L2-7B", "L2-13B"])
    ax.set_ylabel("WikiText2 PPL（越低越好）")
    ax.set_ylim(0, 12)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_zero_shot_figure() -> Path:
    configure_matplotlib()
    path = FIG_DIR / "zero_shot_s07.png"
    models = list(ZERO_SHOT.keys())
    methods = ["Wanda", "SparseGPT", "GS-OWL"]
    x = np.arange(len(models))
    width = 0.24
    colors = ["#A9B0B8", f"#{BLUE}", f"#{ORANGE}"]
    fig, ax = plt.subplots(figsize=(9.3, 3.9), dpi=220)
    for idx, (method, color) in enumerate(zip(methods, colors)):
        vals = [ZERO_SHOT[m][method][2] for m in models]
        bars = ax.bar(x + (idx - 1) * width, vals, width, label=method, color=color, edgecolor="white", linewidth=0.6)
        if method == "GS-OWL":
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, weight="bold")
    ax.set_xticks(x, ["L1-7B", "L1-13B", "L2-7B", "L2-13B"])
    ax.set_ylabel("7 项 zero-shot 平均准确率 / %")
    ax.set_ylim(30, 50)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def export_unified_results() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "ppl_main_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "method", "sparsity_0.5", "sparsity_0.6", "sparsity_0.7", "result_scope"])
        for model, methods in PPL_RESULTS.items():
            writer.writerow([model, "Dense FP16", DENSE_PPL[model], "", "", "literature-reported reference"])
            for method, values in methods.items():
                writer.writerow([model, method, *values, "local experiment logs"])
    with (DATA_DIR / "zero_shot_main_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "method", "sparsity_0.5", "sparsity_0.6", "sparsity_0.7", "metric"])
        for model, methods in ZERO_SHOT.items():
            for method, values in methods.items():
                writer.writerow([model, method, *values, "mean accuracy over 7 tasks"])
    with (DATA_DIR / "strict_ablation_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "sparsity",
                "wanda_uniform",
                "wanda_original_owl",
                "wanda_owl_v9",
                "wsqrtg_uniform",
                "wsqrtg_original_owl",
                "wsqrtg_owl_v9",
            ]
        )
        writer.writerows(ABLATION)


def set_run_font(run, cn=CN_BODY, en=EN_FONT, size=10.5, bold=None, color=INK, italic=None):
    run.font.name = en
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), en)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), en)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_table_borders(table, color="AEB6BF", size="4"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_widths(table, widths):
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)


def format_table_text(table, font_size=8.7):
    for row_idx, row in enumerate(table.rows):
        prevent_row_split(row)
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.05
                for run in paragraph.runs:
                    set_run_font(
                        run,
                        cn=CN_HEAD if row_idx == 0 else CN_BODY,
                        size=font_size,
                        bold=(row_idx == 0),
                        color=WHITE if row_idx == 0 else INK,
                    )
        if row_idx == 0:
            set_repeat_table_header(row)
            for cell in row.cells:
                set_cell_shading(cell, NAVY)
    set_table_borders(table)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def set_doc_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = EN_FONT
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), CN_BODY)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Pt(21)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(1.5)

    for name, size, before, after, color in [
        ("Heading 1", 13.5, 10, 5, NAVY),
        ("Heading 2", 11.5, 7, 3, NAVY),
        ("Heading 3", 10.5, 5, 2, TEAL),
    ]:
        style = styles[name]
        style.font.name = EN_FONT
        style._element.rPr.rFonts.set(qn("w:eastAsia"), CN_HEAD)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.first_line_indent = Pt(0)
        style.paragraph_format.line_spacing = 1.05


def add_text(doc, text, bold_prefix=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=1.5, indent=True):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.first_line_indent = Pt(21) if indent else Pt(0)
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        set_run_font(r1, cn=CN_HEAD, bold=True)
        r2 = p.add_run(text[len(bold_prefix) :])
        set_run_font(r2)
    else:
        run = p.add_run(text)
        set_run_font(run)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(
        run,
        cn=CN_HEAD,
        size={1: 13.5, 2: 11.5, 3: 10.5}[level],
        bold=True,
        color=NAVY if level < 3 else TEAL,
    )
    return p


def add_equation(doc, formula, number):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(formula)
    set_run_font(run, cn=MATH_FONT, en=MATH_FONT, size=11.5, italic=True, color=INK)
    spacer = p.add_run(f"    ({number})")
    set_run_font(spacer, cn=EN_FONT, en=EN_FONT, size=10.5, color=INK)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(text)
    set_run_font(run, cn=CN_BODY, size=9, color=INK)
    p.paragraph_format.keep_with_next = False
    return p


def add_figure(doc, path, caption, width_cm=16.4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption)


def add_table_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, cn=CN_BODY, size=9.5, bold=True, color=INK)
    return p


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if value >= 1000:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def add_main_ppl_table(doc):
    add_table_caption(doc, "表1  不同非结构化稀疏率下的 WikiText2 困惑度（PPL，越低越好）")
    table = doc.add_table(rows=1, cols=5)
    headers = ["模型", "方法", "50%", "60%", "70%"]
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
    for model in PPL_RESULTS:
        row = table.add_row().cells
        row[0].text = model
        row[1].text = "Dense FP16†"
        row[2].text = fmt(DENSE_PPL[model], 2)
        row[3].text = "—"
        row[4].text = "—"
        for method, values in PPL_RESULTS[model].items():
            row = table.add_row().cells
            row[0].text = model
            row[1].text = method
            row[2].text = fmt(values[0])
            row[3].text = fmt(values[1])
            row[4].text = fmt(values[2])
            if method == "GS-OWL":
                for cell in row:
                    set_cell_shading(cell, LIGHT_ORANGE)
    set_table_widths(table, [1900, 2200, 1420, 1420, 1420])
    format_table_text(table, 8.5)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("注：† Dense FP16 为公开文献中同模型的参考值；其余为本地日志解析结果。LLaMA-2-13B 的 Magnitude 同协议结果未找到，以“—”表示。不同历史基线的运行时间与代码版本不完全一致，PPL 仅按相同数据集指标汇总。")
    set_run_font(run, size=8.2, color=MUTED)


def add_zero_shot_table(doc):
    add_table_caption(doc, "表2  七项 zero-shot 任务平均准确率（%，越高越好）")
    table = doc.add_table(rows=1, cols=5)
    for cell, text in zip(table.rows[0].cells, ["模型", "方法", "50%", "60%", "70%"]):
        cell.text = text
    for model, methods in ZERO_SHOT.items():
        for method, values in methods.items():
            row = table.add_row().cells
            row[0].text = model
            row[1].text = method
            row[2].text = "—" if values[0] is None else f"{values[0]:.3f}"
            row[3].text = "—" if values[1] is None else f"{values[1]:.3f}"
            row[4].text = "—" if values[2] is None else f"{values[2]:.3f}"
            if method == "GS-OWL":
                for cell in row:
                    set_cell_shading(cell, LIGHT_ORANGE)
    set_table_widths(table, [1900, 2200, 1420, 1420, 1420])
    format_table_text(table, 8.5)


def add_ablation_table(doc):
    add_table_caption(doc, "表3  剪枝指标与层间分配策略的严格消融（WikiText2 PPL）")
    table = doc.add_table(rows=1, cols=8)
    headers = ["模型", "s", "W-均匀", "W-OWL", "W-V9", "G-均匀", "G-OWL", "G-V9"]
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
    for values in ABLATION:
        row = table.add_row().cells
        row[0].text = "L1" if values[0].startswith("LLaMA-1") else "L2"
        row[1].text = f"{values[1]:.1f}"
        for idx, value in enumerate(values[2:], start=2):
            row[idx].text = f"{value:.3f}"
        set_cell_shading(row[-1], LIGHT_ORANGE)
    set_table_widths(table, [950, 950, 1200, 1200, 1200, 1200, 1200, 1200])
    format_table_text(table, 8.0)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("注：L1/L2 分别表示 LLaMA-1/2-7B，s 为目标稀疏率；W 表示 Wanda 层内指标，G 表示本文 wsqrtg 层内指标；“均匀”表示不进行层间稀疏率重分配。")
    set_run_font(run, size=8.2, color=MUTED)


def add_generalization_table(doc):
    add_table_caption(doc, "表4  GS-OWL 在扩展模型上的 WikiText2 PPL")
    table = doc.add_table(rows=1, cols=4)
    for cell, text in zip(table.rows[0].cells, ["模型", "50%", "60%", "70%"]):
        cell.text = text
    for model, values in FINAL_GENERALIZATION.items():
        row = table.add_row().cells
        row[0].text = model
        row[1].text = fmt(values[0])
        row[2].text = fmt(values[1])
        row[3].text = fmt(values[2])
    set_table_widths(table, [2600, 1900, 1900, 1900])
    format_table_text(table, 8.8)


def add_stability_table(doc):
    add_table_caption(doc, "表5  稳定性与校准样本数量敏感性（稀疏率 60%）")
    table = doc.add_table(rows=1, cols=7)
    headers = ["实验", "模型", "设置", "PPL", "均值±σ", "时间/s", "备注"]
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
    rows = [
        ("种子", "L1-7B", "0/1/2", "8.8426/8.8361/8.8436", "8.8408±0.0041", "170.4/166.2/174.6", "稳定"),
        ("种子", "L2-7B", "0/1/2", "8.1454/8.1419/8.1475", "8.1449±0.0028", "353.9/344.1/342.9", "稳定"),
        ("样本", "L1-7B", "32/64/128/256", "8.8498/8.8486/8.8426/8.8408", "—", "95.3/123.2/175.7/303.3", "收益趋缓"),
        ("样本", "L2-7B", "32/64/128", "8.1458/8.1486/8.1454", "—", "134.5/197.7/355.1", "256 OOM"),
    ]
    for values in rows:
        row = table.add_row().cells
        for cell, value in zip(row, values):
            cell.text = value
    set_table_widths(table, [950, 1050, 1100, 2200, 1400, 2050, 850])
    format_table_text(table, 7.5)


def add_algorithm_box(doc):
    add_table_caption(doc, "算法1  GS-OWL 单次剪枝流程")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "步骤"
    table.rows[0].cells[1].text = "操作"
    steps = [
        ("1", "从校准样本收集层输入能量 X，并读取自回归损失聚合梯度 G。"),
        ("2", "计算层异常证据 O=|W|√X，依据均值倍数阈值获得异常计数 c 与超阈严重度 v。"),
        ("3", "将 c、v 分别映射到层密度，使用 α 锚定融合，并施加核心层保护与中后段上限。"),
        ("4", "在保持全局目标稀疏率的约束下重新归一化，得到每层稀疏率 sₗ。"),
        ("5", "计算层内显著性 M=|W|√|G|，按 sₗ 对每个输出行选择低显著性权重置零。"),
        ("6", "检查实际稀疏率并在 WikiText2 与 zero-shot 任务上评估。"),
    ]
    for number, text in steps:
        row = table.add_row().cells
        row[0].text = number
        row[1].text = text
    set_table_widths(table, [750, 7770])
    format_table_text(table, 8.5)
    for row in table.rows[1:]:
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT


def add_title_block(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run("GS-OWL：面向大语言模型的梯度敏感异常值加权层间剪枝方法")
    set_run_font(run, cn=CN_HEAD, size=17, bold=True, color=INK)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run("作者1，作者2，作者3*")
    set_run_font(run, cn=CN_BODY, size=11, color=INK)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run("（1. 单位名称，城市 邮编；2. 单位名称，城市 邮编）")
    set_run_font(run, cn=CN_BODY, size=9.5, color=MUTED)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run("GS-OWL: Gradient-Sensitive Outlier-Weighted Layerwise Pruning for Large Language Models")
    set_run_font(run, cn=EN_FONT, en=EN_FONT, size=14, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run("AUTHOR One, AUTHOR Two, AUTHOR Three*")
    set_run_font(run, cn=EN_FONT, en=EN_FONT, size=9.5, color=MUTED)


def add_abstracts(doc):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    r1 = p.add_run("摘  要：")
    set_run_font(r1, cn=CN_HEAD, size=10, bold=True)
    abstract = (
        "针对大语言模型一次性非结构化剪枝中“层间稀疏率分配粗糙”和“层内显著性缺少任务敏感信息”两类问题，"
        "提出梯度敏感异常值加权层间剪枝方法 GS-OWL。首先，以权重—激活异常证据为基础，同时建模异常权重数量及其"
        "超阈严重度，通过锚定融合、核心层保护和全局稀疏率重归一化生成非均匀层稀疏率；其次，构造 |W|√|G| 的"
        "梯度敏感显著性指标，在每层预算内保留对自回归损失更敏感的权重。方法无需再训练，也不计算 Hessian。"
        "在 LLaMA-1/2 的 7B、13B 模型上，当全局稀疏率为 60% 时，GS-OWL 相比各模型最优复现基线将 WikiText2 "
        "困惑度降低 5.1%～15.0%；在 70% 高稀疏率下，四个模型七项 zero-shot 任务平均准确率为 44.72%，较 "
        "SparseGPT 高 0.90 个百分点。严格消融表明，OWL-V9 在 Wanda 与梯度指标下均稳定优于原始 OWL，随机种子"
        "实验的 PPL 标准差不超过 0.0041。结果说明，层间异常结构与层内梯度敏感性具有互补作用。"
    )
    r2 = p.add_run(abstract)
    set_run_font(r2, size=9.5)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15

    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run("关键词：")
    set_run_font(r1, cn=CN_HEAD, size=9.5, bold=True)
    r2 = p.add_run("大语言模型；模型剪枝；非结构化稀疏；梯度敏感性；层间稀疏率；OWL")
    set_run_font(r2, size=9.5)

    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.05
    r1 = p.add_run("Abstract: ")
    set_run_font(r1, cn=EN_FONT, en=EN_FONT, size=9.2, bold=True)
    english = (
        "To address coarse layer-wise sparsity allocation and the lack of task-sensitive information in intra-layer saliency, "
        "this paper proposes GS-OWL, a gradient-sensitive outlier-weighted layer-wise pruning method for large language models. "
        "GS-OWL first characterizes both the count and the excess severity of weight-activation outliers, and derives non-uniform "
        "layer sparsities through anchored fusion, core-layer protection, and global sparsity renormalization. It then ranks weights "
        "with |W|sqrt(|G|), preserving parameters that are more sensitive to the autoregressive loss. The method requires neither "
        "retraining nor Hessian computation. On LLaMA-1/2 7B and 13B models at 60% sparsity, GS-OWL reduces WikiText2 perplexity by "
        "5.1%-15.0% against the strongest reproduced baseline for each model. At 70% sparsity, its mean accuracy over seven zero-shot "
        "tasks reaches 44.72%, 0.90 percentage points above SparseGPT. Strict ablations further confirm the complementary effects of "
        "the proposed layer allocation and gradient-sensitive metric."
    )
    r2 = p.add_run(english)
    set_run_font(r2, cn=EN_FONT, en=EN_FONT, size=8.8)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run("Key words: ")
    set_run_font(r1, cn=EN_FONT, en=EN_FONT, size=8.8, bold=True)
    r2 = p.add_run("large language model; model pruning; unstructured sparsity; gradient sensitivity; layer-wise sparsity; OWL")
    set_run_font(r2, cn=EN_FONT, en=EN_FONT, size=8.8)

    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run("中图分类号：TP391.1    文献标志码：A    DOI：待编辑部分配")
    set_run_font(run, size=8.8, color=MUTED)


def add_front_note(doc):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run("基金项目：待补充；作者简介：待补充；通信作者：待补充（E-mail：待补充）。")
    set_run_font(run, size=8.5, color=MUTED)
    p.add_run().add_break(WD_BREAK.LINE)
    run = p.add_run("投稿前数据核验提示：Dense FP16 为文献参考值；LLaMA-2-13B 的 Magnitude 结果缺失；Qwen2.5-7B 因当前困惑度评测兼容性异常未纳入正文。")
    set_run_font(run, size=8.5, color=ORANGE)


def build_document(figures):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.65)
    section.bottom_margin = Cm(1.65)
    section.left_margin = Cm(1.75)
    section.right_margin = Cm(1.75)
    section.header_distance = Cm(0.65)
    section.footer_distance = Cm(0.65)

    set_doc_styles(doc)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header.paragraph_format.space_after = Pt(0)
    hr = header.add_run("计算机工程与应用 · 投稿初稿")
    set_run_font(hr, cn=CN_BODY, size=8.5, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run("GS-OWL    ")
    set_run_font(fr, cn=EN_FONT, en=EN_FONT, size=8, color=MUTED)
    add_page_number(footer)

    add_title_block(doc)
    add_abstracts(doc)
    add_front_note(doc)

    add_heading(doc, "0  引言", 1)
    add_text(
        doc,
        "大语言模型（Large Language Model，LLM）依靠参数规模和数据规模获得了显著的生成与推理能力[1-3]，"
        "但数十亿乃至数百亿参数也带来了显存占用、存储和部署成本。模型剪枝通过移除冗余权重构造稀疏网络，"
        "是量化之外的重要压缩路线[4-7]。与需要再训练的传统剪枝不同，一次性剪枝只使用少量校准样本即可完成压缩，"
        "更适合难以重新训练的大语言模型。",
    )
    add_text(
        doc,
        "现有一次性剪枝主要回答两个问题：一是“同一层内保留哪些权重”，二是“不同 Transformer 层应分配多少稀疏预算”。"
        "SparseGPT[8]利用近似二阶信息逐层补偿误差，精度较高但计算和显存开销较大；Wanda[9]以权重幅值和输入激活"
        "能量的乘积衡量显著性，避免 Hessian 与重训练。OWL[10]进一步指出，异常权重在层间分布不均，统一稀疏率会"
        "过度破坏敏感层，因此依据异常比例调整层密度。RIA[11]与 Pruner-Zero[12]则分别从相对重要性和自动搜索符号"
        "指标的角度改进层内排序。",
    )
    add_text(
        doc,
        "然而，原始 OWL 主要依据超过均值倍数阈值的异常权重“数量”分配密度，未区分刚刚越过阈值和远高于阈值的权重，"
        "当多个层的异常比例接近时，分配信号容易退化。另一方面，激活能量描述输入分布，却不直接反映权重对语言建模损失"
        "的局部敏感性。单独使用权重—激活指标可能保留高响应但任务梯度较小的参数，也可能误删幅值一般却对损失敏感的参数。",
    )
    add_text(doc, "针对上述问题，本文提出 GS-OWL。主要贡献如下：", indent=False)
    contributions = [
        "提出梯度敏感层内显著性 wsqrtg，以 |W|√|G| 同时编码权重承载能力和自回归损失敏感性，保持线性复杂度且无需 Hessian。",
        "提出 OWL-V9 层间分配策略，将异常数量与超阈严重度锚定融合，并通过核心层保护、尾部上限和全局重归一化保证分配稳定与目标稀疏率。",
        "在 LLaMA-1/2、Mistral 和 OPT 等模型上完成多稀疏率实验，并通过 2×3 严格消融、360 组参数搜索、随机种子、校准样本数及层分布分析验证两个创新点。",
    ]
    for idx, item in enumerate(contributions, 1):
        add_text(doc, f"（{idx}）{item}", indent=False, after=1)

    add_heading(doc, "1  相关工作", 1)
    add_heading(doc, "1.1  大语言模型一次性剪枝", 2)
    add_text(
        doc,
        "早期剪枝研究依据权重幅值或二阶曲率估计参数重要性[4-7]。在 LLM 场景中，SparseGPT[8]使用逐层稀疏回归与"
        "Hessian 近似，在较高稀疏率下保持较低 PPL；Wanda[9]以 |W|√X 形成轻量级显著性，证明无需权重更新也能获得"
        "有竞争力的结果。LLM-Pruner[13]和 SliceGPT[14]探索结构化维度删除，能够带来真实加速，但通常具有更大的结构"
        "扰动。本文聚焦非结构化剪枝，其优点是稀疏率控制细粒度、对模型架构侵入较小；其真实推理加速仍依赖稀疏算子支持。",
    )
    add_heading(doc, "1.2  层间稀疏率与自动指标", 2)
    add_text(
        doc,
        "OWL[10]观察到 LLM 的异常权重分布具有显著层间差异，并将异常比例高的层分配更高密度。该思想把层内排序器和"
        "层间预算器解耦，因此可以与 Wanda、SparseGPT 或其他指标组合。RIA[11]通过相对重要性缓解行列尺度不均；"
        "Pruner-Zero[12]利用进化搜索自动发现符号剪枝指标。与上述工作相比，本文不搜索更复杂表达式，而是将层内排序"
        "明确约束为权重与梯度的几何组合，同时针对 OWL 只统计异常数量的局限引入严重度信号。",
    )
    add_heading(doc, "1.3  量化与其他压缩方法", 2)
    add_text(
        doc,
        "GPTQ[15]、AWQ[16]和 LLM.int8()[17]从数值精度角度降低模型存储和计算成本，通常可与剪枝叠加。本文关注"
        "FP16 权重上的一次性非结构化剪枝，不讨论量化误差和稀疏—量化联合优化。近期动态层剪枝与旋转去噪方法[23-26]"
        "表明，模型压缩正从单一静态指标转向结构先验、数据敏感性和误差控制的联合建模，这也构成 GS-OWL 的设计动机。",
    )

    add_heading(doc, "2  问题定义与预备知识", 1)
    add_text(
        doc,
        "设预训练 Transformer 包含 L 个待剪枝线性层组，第 l 层权重为 Wₗ。给定全局目标稀疏率 s，目标是在不更新"
        "剩余权重的前提下生成二值掩码 Bₗ，使所有层被置零权重比例接近 s，并尽可能减小剪枝模型在语言建模损失与下游任务"
        "上的性能退化。非均匀层分配进一步要求为每层确定 sₗ，并满足参数加权后的全局约束。",
    )
    add_equation(doc, "min  L(f(W ⊙ B); D),   s.t.  Σₗ ||1-Bₗ||₀ / Σₗ |Wₗ| = s", 1)
    add_text(
        doc,
        "Wanda 在校准样本上累计每个输入通道的激活平方和 X，并使用 |W|√X 排序。原始 OWL 以该指标的均值倍数为阈值，"
        "统计异常权重比例，再映射为层密度。GS-OWL 保留该异常证据作为层间敏感性来源，从而使消融变量清晰；真正执行置零时"
        "则使用梯度敏感指标。",
    )

    add_heading(doc, "3  GS-OWL 方法", 1)
    add_figure(doc, figures["overview"], "图1  GS-OWL 方法总体流程", 16.5)
    add_heading(doc, "3.1  梯度敏感层内显著性", 2)
    add_text(
        doc,
        "对于权重 wᵢⱼ，幅值反映其数值承载能力，一阶梯度 gᵢⱼ 反映当前校准目标对该权重的局部敏感性。若直接使用"
        "|W||G|，少量极大梯度可能主导排序；若只使用 |G|，尺度较小但梯度噪声较大的权重又可能被过度保护。本文对梯度"
        "幅值进行平方根压缩，定义 wsqrtg 指标：",
    )
    add_equation(doc, "Mᵢⱼ = |Wᵢⱼ| · √(|Gᵢⱼ| + ε)", 2)
    add_text(
        doc,
        "式中 G 为在校准语料上对自回归语言建模损失求得并按样本聚合的梯度，ε 为数值稳定项。平方根既保持梯度排序信息，"
        "又降低重尾梯度对阈值的放大。对每个输出行按 M 从小到大排序，根据该层预算 sₗ 置零最低分权重。",
    )
    add_heading(doc, "3.2  异常数量与超阈严重度", 2)
    add_text(
        doc,
        "为了与原始 OWL 保持一致，层间证据仍由权重和输入激活构造。对第 l 层各线性模块计算 Oₗ=|Wₗ|√Xₗ，并以均值"
        "倍数定义阈值 τₗ，其中 Hₘ 控制异常判定严格程度：",
    )
    add_equation(doc, "τₗ = Hₘ · mean(Oₗ),    cₗ = |{Oₗ > τₗ}| / |Oₗ|", 3)
    add_text(
        doc,
        "仅有 cₗ 时，两个异常数量相近但强度差异显著的层会获得近似密度。为此，本文定义异常元素相对阈值的平均超额 eₗ，"
        "并通过 log1p 抑制极端值，形成严重度加权比例 vₗ：",
    )
    add_equation(doc, "eₗ = E[(Oₗ-τₗ)/(τₗ+ε) | Oₗ>τₗ],    vₗ = cₗ[1+log(1+eₗ)]", 4)
    add_heading(doc, "3.3  锚定密度融合与约束归一化", 2)
    add_text(
        doc,
        "对任意层信号 rₗ，先执行最小—最大归一化 zₗ，再以目标稀疏率 s 为中心、以 λ 为最大调整幅度映射为层稀疏率，"
        "并转换为密度 D(rₗ)。异常多的层获得更高密度，即更低稀疏率。计数信号稳定但区分度有限，严重度信号区分度高但"
        "可能受极端值影响，因此采用 α 锚定融合：",
    )
    add_equation(doc, "d̃ₗ = (1-α)D(cₗ) + αD(vₗ),    s̃ₗ = 1-d̃ₗ", 5)
    add_text(
        doc,
        "在实现中，对 LLaMA 类模型的第 4～10 层施加核心密度下界，使其密度不低于计数分支；对第 16～30 层施加密度"
        "上限，抑制严重度噪声导致的尾部过度保护。最后在边界约束内迭代平移所有层密度，使参数加权平均稀疏率回到目标 s。"
        "这一过程不会改变全局剪枝预算。",
    )
    add_equation(doc, "Σₗ Nₗ(1-dₗ) / Σₗ Nₗ = s,    dₗ ∈ [1-s-λ, 1-s+λ]", 6)
    add_algorithm_box(doc)
    add_heading(doc, "3.4  复杂度分析", 2)
    add_text(
        doc,
        "设待剪枝参数量为 P。异常证据、梯度显著性和掩码选择均为逐元素操作，时间复杂度为 O(P)，不构造 d×d Hessian；"
        "附加存储主要为聚合梯度与临时指标，最坏为 O(P)。与 Wanda 相比，GS-OWL 增加一次梯度获取或梯度文件读取；与"
        "SparseGPT 相比，不进行矩阵逆近似和逐列误差补偿。需要强调的是，非结构化稀疏本身不会自动带来硬件加速，实际吞吐"
        "提升依赖稀疏内核和部署后端。",
    )

    add_heading(doc, "4  实验设置", 1)
    add_heading(doc, "4.1  模型、数据集与指标", 2)
    add_text(
        doc,
        "主实验采用 LLaMA-1-7B、LLaMA-1-13B、LLaMA-2-7B 和 LLaMA-2-13B，并使用 LLaMA-1-30B、Mistral-7B "
        "和 OPT-1.3B 检查规模与架构泛化性。校准阶段从 C4 采样 128 个序列，主实验随机种子为 0。语言建模性能使用"
        "WikiText2[18] PPL；下游能力使用 BoolQ[19]、RTE、HellaSwag[20]、ARC-Challenge、ARC-Easy、"
        "WinoGrande[21]和 OpenBookQA[22]七项 zero-shot 准确率及其算术平均值。",
    )
    add_heading(doc, "4.2  对比方法与参数", 2)
    add_text(
        doc,
        "对比方法包括 Dense FP16、Magnitude、Wanda[9]、SparseGPT[8]、RIA[11]、Pruner-Zero[12]与 GBLM。"
        "为验证两个创新点，额外构造 Wanda/梯度指标与均匀分配/原始 OWL/OWL-V9 的 2×3 组合。所有剪枝均为非结构化"
        "一次性剪枝，目标稀疏率为 50%、60%和 70%。参数网格设置 Hₘ∈{5,6,7}，λ∈{0.02,0.04,…,0.20}，"
        "α∈{0.10,0.15,0.20,0.25}，每个模型共 360 组。主表采用每个稀疏率 PPL 最低的组合；zero-shot 主结果报告"
        "PPL 排名第 1 的参数，而 PPL 前 10 组的 zero-shot 最优值仅用于附加分析，避免直接以 zero-shot 测试任务选参。",
    )
    add_heading(doc, "4.3  实现与可比性说明", 2)
    add_text(
        doc,
        "实验基于 PyTorch 和 Transformers，在 A40、A100 与 L20 GPU 上完成。由于历史基线和最终方法的运行时间、"
        "Transformers 小版本及部分模型路径不同，本文仅在同一 A40 环境下比较 Wanda-OWL-V9 与 GS-OWL 的剪枝时间；"
        "跨硬件时间不用于方法效率排序。Dense FP16 数值来自公开文献的同模型参考结果，Magnitude 的 LLaMA-2-13B "
        "同协议日志缺失。当前本地结果中，原始 OWL 与 OWL-V9 尚无同协议 zero-shot 文件，因此表2只汇总可核验的"
        "经典基线与最终方法，层间分配策略的 zero-shot 单变量收益留待投稿前补充。Qwen2.5-7B 的当前 WikiText2 "
        "评测出现异常大 PPL，初步判断与 tokenizer/评测适配有关，故不纳入正文结论。",
    )

    add_heading(doc, "5  实验结果与分析", 1)
    add_heading(doc, "5.1  主性能比较", 2)
    add_main_ppl_table(doc)
    add_figure(doc, figures["ppl"], "图2  60% 稀疏率下主要方法的 WikiText2 PPL", 15.8)
    add_text(
        doc,
        "由表1可见，50% 稀疏率时各强基线差距较小，GS-OWL 在 LLaMA-1-7B 上以 0.0186 PPL 略逊于 Pruner-Zero，"
        "但在其余三个模型上取得最低 PPL。稀疏率升至 60% 后，GS-OWL 在四个模型上分别达到 8.8426、7.0694、"
        "8.1455 和 6.5466；相对每个模型的最优复现基线，PPL 分别下降约 11.0%、8.0%、15.0%和 5.1%。"
        "这说明在中高稀疏率下，统一层预算和单一激活显著性的误差被放大，层间分配与梯度敏感性更有价值。",
    )
    add_text(
        doc,
        "在 70% 稀疏率下，GS-OWL 在 LLaMA-1-7B、LLaMA-1-13B 和 LLaMA-2-13B 上明显优于 SparseGPT，"
        "PPL 分别降低约 20.8%、38.8%和 31.9%；但在 LLaMA-2-7B 上为 25.5248，略高于 SparseGPT 的 24.2518。"
        "该例外表明，梯度一阶信息不能在所有模型上替代二阶误差补偿，本文方法在极高稀疏率下仍存在模型依赖性。",
    )
    add_heading(doc, "5.2  Zero-shot 下游能力", 2)
    add_zero_shot_table(doc)
    add_figure(doc, figures["zero"], "图3  70% 稀疏率下七项 zero-shot 平均准确率", 15.8)
    add_text(
        doc,
        "表2显示，50%和 60% 稀疏率下，最低 PPL 参数并不总对应最高 zero-shot 均值，GS-OWL 与 SparseGPT、RIA 的"
        "差距通常在 0～2.2 个百分点内。这一现象说明语言建模困惑度与离散选择题任务并非完全一致，也验证了本文将"
        "PPL 前 10 组仅作为附加分析、而不以 zero-shot 测试集直接调参的必要性。",
    )
    add_text(
        doc,
        "当稀疏率达到 70% 时，GS-OWL 在四个主模型上的 zero-shot 均值分别为 43.326、46.781、43.511 和 47.243，"
        "平均为 44.72%，较 SparseGPT 的四模型平均值高 0.90 个百分点，并在四个模型上均取得更高平均准确率。"
        "这表明 GS-OWL 虽未在每个模型上获得最低 PPL，但其保留的梯度敏感权重在高压缩条件下有助于维持通用任务能力。",
    )
    add_heading(doc, "5.3  两个创新点的严格消融", 2)
    add_ablation_table(doc)
    add_text(
        doc,
        "固定 Wanda 指标时，OWL-V9 在两个模型、三个稀疏率下均优于原始 OWL；固定 wsqrtg 指标时同样 6/6 获胜，"
        "证明严重度加权与约束分配的收益并非由层内指标替换造成。另一方面，固定 OWL-V9 后，wsqrtg 在 6 个设置中的"
        "5 个优于 Wanda，唯一例外为 LLaMA-2-7B 的 70% 稀疏率（25.5227 对 19.4580）。因此，两个创新点总体互补，"
        "但梯度指标在个别极端稀疏设置下会受到一阶梯度噪声影响。",
    )
    add_heading(doc, "5.4  层间分布可视化", 2)
    add_figure(doc, figures["layer"], "图4  60% 全局稀疏率下的层间稀疏率分布（绿色区域为核心层，橙色区域为中后段约束区）", 16.2)
    add_text(
        doc,
        "图4显示，两种模型均呈现“前中层较低稀疏率、后层较高稀疏率”的非均匀结构。OWL-V9 与原始 OWL 的总体趋势"
        "一致，说明计数锚定避免了分配方向突变；同时，严重度分支在第 0～10 层形成更细粒度修正，并通过核心层下界阻止"
        "敏感层被过剪。所有层调整后仍满足 60% 全局预算。",
    )
    add_heading(doc, "5.5  参数、稳定性与校准成本", 2)
    add_stability_table(doc)
    add_text(
        doc,
        "随机种子实验中，LLaMA-1-7B 和 LLaMA-2-7B 的 PPL 标准差分别为 0.0041 和 0.0028，说明在固定参数下"
        "结果稳定。校准样本从 32 增加到 128 时，LLaMA-1-7B 的 PPL 小幅改善 0.0072，继续增加到 256 的收益仅"
        "0.0018，而剪枝时间从 95.3 s 增至 303.3 s。LLaMA-2-7B 的 32～128 样本结果几乎不变，256 样本在当前"
        "显存配置下 OOM。因此 128 个样本是精度、时间与显存之间较合理的折中。",
    )
    add_text(
        doc,
        "360 组参数搜索显示，λ 对 PPL 和层间分布影响最大，Hₘ 次之，α 的影响相对较弱。随着目标稀疏率提高，较大的 λ "
        "通常更有利，因为高稀疏率需要更强的层间差异来保护敏感层；α 的最优值多位于 0.10～0.20，支持“计数为锚、"
        "严重度为修正”的设计，而不是让严重度完全主导密度。",
    )
    heading = add_heading(doc, "5.6  泛化性与效率", 2)
    heading.paragraph_format.page_break_before = True
    add_generalization_table(doc)
    add_text(
        doc,
        "GS-OWL 在 LLaMA-1-30B 上的 PPL 为 4.9902/5.9740/9.0048，表明方法可扩展到更大参数规模；在 Mistral-7B "
        "上同样保持可用结果。OPT-1.3B 在 70% 稀疏率下退化明显，说明较小模型的冗余度有限，不能直接沿用大模型的高"
        "稀疏率结论。实际稀疏率与目标值的绝对误差均不超过 0.0007。",
    )
    add_text(
        doc,
        "在相同 A40 环境和 LLaMA-1-7B 上，Wanda-OWL-V9 三个稀疏率的平均剪枝时间为 211.36 s，GS-OWL 为"
        "240.76 s，增加约 13.9%。该开销主要来自梯度指标读取与计算，仍显著低于需要二阶矩阵处理的算法复杂度。由于"
        "实验未接入专用稀疏推理后端，本文不报告推理吞吐或延迟加速。",
    )

    add_heading(doc, "6  讨论与局限性", 1)
    add_text(
        doc,
        "第一，当前核心层与中后段约束区使用固定层编号，这一工程先验在 LLaMA 类结构上有效，但对层数差异较大的模型未必"
        "最优。后续可改为按相对深度或基于层敏感性自动确定区间。第二，梯度由校准数据决定，领域偏移可能改变排序；可研究"
        "多域梯度融合或 Fisher 信息的低成本近似。第三，参数网格按模型和稀疏率分别搜索，展示的是方法可达到的性能上界，"
        "仍需补充“在 7B 上选参、向 13B/30B 迁移”的固定参数实验，以进一步排除逐模型调参收益。",
    )
    add_text(
        doc,
        "第四，历史基线日志的运行日期、软件小版本和硬件环境并不完全相同。本文通过同一代码路径完成 Wanda/OWL/V9 与"
        "wsqrtg 的严格消融，但 Dense、Magnitude、SparseGPT 等统一主表仍应在投稿前复核 tokenizer、数据 split 和"
        "评测脚本版本。第五，非结构化稀疏只减少非零参数数量，不保证通用 GPU 上获得线性加速；真实部署价值需要在支持"
        "2:4 或块稀疏的后端中进一步验证。",
    )

    add_heading(doc, "7  结论", 1)
    add_text(
        doc,
        "本文提出 GS-OWL，一种面向大语言模型的一次性非结构化剪枝方法。方法以异常数量与超阈严重度共同决定层间预算，"
        "并以 |W|√|G| 在层内保留梯度敏感权重。在 LLaMA-1/2 7B、13B 以及扩展模型上的实验表明，GS-OWL 在"
        "60%～70% 稀疏率下能够显著缓解 PPL 退化，并在 70% 稀疏率下保持更好的 zero-shot 平均准确率。严格消融进一步"
        "证明 OWL-V9 层分配和 wsqrtg 指标总体上具有互补性。后续将研究相对深度约束、跨模型参数迁移和结构化稀疏后端，"
        "把算法层面的非零权重减少转化为可复现的端到端推理收益。",
    )

    add_heading(doc, "参考文献", 1)
    for idx, reference in enumerate(REFERENCES, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.first_line_indent = Pt(-18)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(f"[{idx}] {reference}")
        set_run_font(run, cn=CN_BODY, en=EN_FONT, size=8.5, color=INK)

    doc.core_properties.title = "GS-OWL：面向大语言模型的梯度敏感异常值加权层间剪枝方法"
    doc.core_properties.subject = "《计算机工程与应用》投稿初稿"
    doc.core_properties.keywords = "大语言模型; 模型剪枝; 非结构化稀疏; 梯度敏感性; OWL"
    doc.core_properties.author = "待补充"
    doc.save(DOCX_PATH)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    export_unified_results()
    figures = {
        "overview": make_overview_figure(),
        "layer": make_layer_figure(),
        "ppl": make_main_result_figure(),
        "zero": make_zero_shot_figure(),
    }
    build_document(figures)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
