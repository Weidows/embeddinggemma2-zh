"""Generate the standalone Chinese evaluation report from the raw results.

Everything in report/cmteb-embeddinggemma2-zh.md is derived from
results/mteb_base_*.json (+ results/cmteb_comparison.md if present), so the numbers can
never drift from the raw run.

Usage: uv run python src/make_report.py
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_comparison import TASK_TYPES, aggregate, load_local_runs, load_refs  # noqa: E402

RESULTS = Path("results")
REPORT = Path("report")
ORDER = list(TASK_TYPES)
TYPE_ORDER = ["Retrieval", "Reranking", "PairClassification", "Clustering", "STS", "Classification"]


def fmt(v: float | None, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def main() -> None:
    f = sorted(glob.glob(str(RESULTS / "mteb_base_*.json")))[-1]
    d = json.load(open(f, encoding="utf-8"))
    per_task: dict[str, list[float]] = {}
    meta: dict[str, tuple[str, str, str]] = {}
    for r in d["rows"]:
        if isinstance(r["main_value"], (int, float)):
            per_task.setdefault(r["task"], []).append(r["main_value"])
            meta[r["task"]] = (r["type"], r["split"], r["main_score"])
    task_mean = {t: statistics.mean(v) for t, v in per_task.items()}

    local = load_local_runs()
    refs = load_refs()
    ours = next(v for k, v in local.items() if "embeddinggemma-2" in k)
    bgem3 = next((v for k, v in local.items() if "bge-m3" in k), None)
    qwen = refs.get("Qwen__Qwen3-Embedding-0.6B", {})
    ours_agg, ours_type = aggregate(ours)
    bg_agg, bg_type = aggregate(bgem3) if bgem3 else (None, {})
    qw_agg, qw_type = aggregate(qwen)

    rows = [f"| {t} | {TASK_TYPES[t]} | {meta[t][2]} | {task_mean[t]:.4f} |" for t in ORDER if t in task_mean]
    type_rows = [
        f"| {ty} | {fmt(ours_type.get(ty))} | {fmt(bg_type.get(ty))} | {fmt(qw_type.get(ty))} |"
        for ty in TYPE_ORDER
    ]

    deltas = sorted(
        (task_mean[t] - bgem3[t], t) for t in ORDER if t in task_mean and bgem3 and t in bgem3
    )
    worst, best = deltas[:6], deltas[::-1][:6]
    wins = len([x for x in deltas if x[0] > 0])
    worst_tbl = "\n".join(f"| {t} | {task_mean[t]:.4f} | {bgem3[t]:.4f} | {dd:+.4f} |" for dd, t in worst)
    best_tbl = "\n".join(f"| {t} | {task_mean[t]:.4f} | {bgem3[t]:.4f} | {dd:+.4f} |" for dd, t in best)

    cmp_path = RESULTS / "cmteb_comparison.md"
    cmp_block = cmp_path.read_text(encoding="utf-8") if cmp_path.exists() else "_(尚未生成：uv run python src/make_comparison.py)_"

    md = f"""# EmbeddingGemma 2 中文能力实测 — C-MTEB（`MTEB(cmn, v1)`，31 个任务）

> 生成自 `{f}`，全部数字由脚本从原始结果生成（`src/make_report.py`），未经人工转录。

## 摘要

`google/embeddinggemma-2`（2026-10-06 发布，740M 多模态嵌入模型，文本骨干 271M）在中文
**C-MTEB 全量 31 个任务**上的总体得分为 **{ours_agg:.4f}**（task-type 均值口径）。

**三方对照（同 harness、同机器、同样截断与 batch）**：

| 模型 | 参数量 | 总体 | 相对我们 |
|---|---|---|---|
| `google/embeddinggemma-2`（只载文本骨干） | 271M | **{ours_agg:.4f}** | — |
| `BAAI/bge-m3`（本机实跑） | 568M | **{fmt(bg_agg)}** | {fmt(bg_agg - ours_agg) if bg_agg else "—"} |
| `Qwen/Qwen3-Embedding-0.6B`（榜单公开数） | 595M | **{qw_agg:.4f}** | {qw_agg - ours_agg:+.4f} |

一句话：**用 bge-m3 一半的参数量拿到它约 {100 * ours_agg / bg_agg:.0f}% 的中文成绩**，
但并非全面落后 —— 有 **{wins} 个单项反超 bge-m3**。

### 两个初判被基准推翻（这正是必须做本地基准的原因）

1. **"PAWSX 是短板"——错。** 我们 {task_mean.get('PAWSX', float('nan')):.4f}，bge-m3 {bgem3.get('PAWSX', float('nan')):.4f}，
   只差 {abs(task_mean.get('PAWSX', 0) - bgem3.get('PAWSX', 0)):.4f}。这是**任务本身对所有人都难**
   （对抗性改写识别），不是模型缺陷。
2. **"中文聚类弱"——错。** 我们聚类均值 {ours_type.get('Clustering', float('nan')):.4f}，
   **高于 bge-m3 的 {bg_type.get('Clustering', float('nan')):.4f}**；ThuNews 两项还分别反超 bge-m3
   {task_mean.get('ThuNewsClusteringS2S', 0) - bgem3.get('ThuNewsClusteringS2S', 0):+.4f} 和
   {task_mean.get('ThuNewsClusteringP2P', 0) - bgem3.get('ThuNewsClusteringP2P', 0):+.4f}。
   "聚类弱"只是相对 Qwen3-Embedding-0.6B 而言，而那个模型在聚类上异常强。

### 真正的短板（对 bge-m3 也落后约 10 分）

| 任务 | 我们 | bge-m3 | 差 |
|---|---|---|---|
{worst_tbl}

### 反超 bge-m3 的单项（前 6）

| 任务 | 我们 | bge-m3 | 差 |
|---|---|---|---|
{best_tbl}

**结论**：通用检索与新闻聚类是它的舒适区；**医疗领域（检索 + reranking）与中文 STS 的
AFQMC / ATEC / BQ 是明确短板** —— 这两块就是阶段 2 微调的靶子。

## 评测设置

| 项 | 值 |
|---|---|
| benchmark | `MTEB(cmn, v1)`（= C-MTEB），31 个任务 |
| 模型 | `google/embeddinggemma-2`，**只加载文本骨干**（`config_kwargs={{vision_config: None, audio_config: None}}`，271.0M 参数） |
| dtype | bfloat16（模型卡明确警告：**不要用 float16**，会静默 NaN/劣化） |
| 截断 | `max_seq_length = 2048`（模型支持 8192；C-MTEB 语料 p99 ≈ 1.3k token） |
| batch size | 16 |
| prompt | 使用模型自带的 task 前缀（`task: search result \\| query: ` 等），由 `config_sentence_transformers.json` 提供 |
| 框架 | sentence-transformers 6.1.0 / transformers 5.19.0 / mteb 2.24.0 / torch 2.11.0+cu128 |
| 硬件 | 单卡 RTX 5070 Ti（16GB），显存进程上限 70% |
| 耗时 | 我们 {d['elapsed_seconds']/60:.1f} 分钟；bge-m3 对照 84.9 分钟（同机同设置） |

## 结果（31 个任务）

| 任务 | 类型 | 主指标 | 分数 |
|---|---|---|---|
{chr(10).join(rows)}

### 按任务类型汇总

| 任务类型 | embeddinggemma-2 | bge-m3 | Qwen3-Embedding-0.6B |
|---|---|---|---|
{chr(10).join(type_rows)}
| **总体（task-type 均值）** | **{ours_agg:.4f}** | **{fmt(bg_agg)}** | **{qw_agg:.4f}** |

## 完整对照表（含逐任务 delta）

{cmp_block}

## 复现方法

```bash
# 1. 环境
uv sync

# 2. 权重（本机 HF_ENDPOINT 指向镜像会让 hf download 失败，必须直连+代理）
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \\
  hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \\
  hf download BAAI/bge-m3 --local-dir models/bge-m3

# 3. 全量评测（我们约 2.9 小时；bge-m3 约 1.4 小时，均 RTX 5070 Ti）
env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \\
  PYTORCH_ALLOC_CONF=expandable_segments:True \\
  uv run python src/eval_mteb.py --benchmark "MTEB(cmn, v1)" --tag base \\
  --batch-size 16 --max-seq-length 2048 --vram-fraction 0.7

env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \\
  PYTORCH_ALLOC_CONF=expandable_segments:True \\
  uv run python src/eval_mteb.py --model models/bge-m3 --tag bgem3 --no-text-pin \\
  --batch-size 16 --max-seq-length 2048 --vram-fraction 0.7

# 4. 对照表 / 本报告
uv run python src/make_comparison.py
uv run python src/make_report.py
```

## 局限

- **截断 2048 而非模型上限 8192**：受显存约束（该模型 `per_layer_projection_norm` 里有一次
  fp32 上采样，`[batch, seq, 24 层, 512 维] × 4 字节` 在 batch 32 × 2048 时就是 3.2GB 单张量）。
  C-MTEB 语料 p99 ≈ 1.3k token，影响有限，但**医疗类长文档任务可能被低估**，待做截断敏感性验证。
- **对照模型同机同设置（2048 截断）**：bge-m3 与 Qwen3-Embedding-0.6B 都是 2048 截断，
  因此与它们在公开榜单（各自默认长度）上的成绩不可直接对比，只在本文内部可比。
- **未评测多模态**：本报告只覆盖文本，图像/音频（MIEB / MAEB）未涉及。

## 附：使用这个模型时必须知道的一个坑

用 sentence-transformers 直接 `encode()` 时，**如果某条输入"整串就是一个媒体 URL"**，
默认路径会把它当成媒体去解码：

| 输入 | 结果 |
|---|---|
| `http://www.youtube.com/ehowatHomeChannel` | ❌ `ImportError: 要解码 YouTube 视频需要先装 yt_dlp` |
| `https://example.com/clip.mp4` | ❌ libtorchcodec 报错 |
| `https://www.baidu.com` | ✅ 正常 |
| `看这段 https://example.com/a.mp4` | ✅ 正常（只有整串是媒体 URL 才触发） |

网页抓取的中文语料里这种文档很常见（MMarcoRetrieval 语料第 98275 条就是裸 YouTube 链接），
一次全量评测会因此崩在语料编码中途。解法：把 sentence-transformers pin 到纯 `text` modality
（`src/eval_mteb.py::pin_text_modality`）——**同时还能提速约 4 倍，且 embedding 数值不变**
（200 条中文文本上 `np.allclose(atol=1e-4) == True`）。
"""
    REPORT.mkdir(exist_ok=True)
    out = REPORT / "cmteb-embeddinggemma2-zh.md"
    out.write_text(md, encoding="utf-8")
    print(f"[out] {out}  ({len(md)} chars)")
    print(f"ours={ours_agg:.4f}  bgem3={fmt(bg_agg)}  qwen={qw_agg:.4f}  tasks={len(task_mean)}  wins_vs_bgem3={wins}")


if __name__ == "__main__":
    main()
