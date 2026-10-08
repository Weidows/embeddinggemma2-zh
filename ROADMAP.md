# ROADMAP — embeddinggemma2-zh

**北极星方向**：让中文用户能用上 `google/embeddinggemma-2`（740M 多模态嵌入）——
先用**可信的中文数字**回答"它中文到底行不行"，再用**中文微调**把差距补上，
产出可被第三方独立复现的评测表与模型权重。

> 本文件是唯一真源：步骤、为什么、运行记录、验收成果都写这里。状态：✅ 完成 / ⏳ 进行中 / ⬜ 未开始

---

## 背景（2026-10-08 核实）

- 模型 `google/embeddinggemma-2`：2026-10-06 发布，740M（270M 文本 + 170M 视觉 + 300M 音频），
  统一 768 维 / MRL 可截断到 512/256/128，8K 上下文，Apache 2.0。
- 生态现状：GGUF(+mmproj)、MLX、ONNX、LiteRT-LM、CoreML、WebGPU、vLLM 支持**均已被人做完**。
- **真空**：截至今日用 `filter=base_model:google/embeddinggemma-2` 拉出全部 56 个衍生仓库 → **零中文**；
  中文圈只有转述贴，**没有任何 C-MTEB / MTEB-zh 实测数字**。
- 模型卡警告：**不要用 float16**（激活范围超 fp16，会静默 NaN/劣化）。

---

## 阶段 1：中文评测（出数字）

| # | 步骤 | 为什么 | 运行记录 | 验收成果（可独立验证） | 状态 |
|---|---|---|---|---|---|
| 1.1 | uv 项目 + cu128 torch（sm_120） | 5070Ti 必须 cu128+，否则静默退回 CPU | `uv sync` → torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / transformers 5.19.0 / sentence-transformers 6.1.0 / mteb 2.24.0 | `uv run python -c "import torch;print(torch.__version__,torch.cuda.is_available())"` → `2.11.0+cu128 True` | ✅ |
| 1.2 | 跑通模型加载（text-only，关掉 vision/audio 编码器） | 740M 全量加载浪费显存；文本评测只需 270M | `src/sanity_check.py`：params=**271.0M**，dim=768，模型卡示例 cos(query,doc)=**0.8716** | 上述数字与模型卡一致；`[model] params=271.0M` | ✅ |
| 1.3 | 对齐 sentence-transformers 的 `prompt_name` 前缀 | 模型卡：漏掉 task 前缀会明显掉分，评测不公平 | cos(SearchQuery, Clustering)=0.8171、cos(SearchQuery, 无前缀)=0.8850（都 <1 → 前缀确实生效）；MRL 256 维重归一后 \|v\|=0.9974 | 同句不同前缀分数必须 <1.0 | ✅ |
| 1.4 | 跑 MTEB-zh 任务集（`MTEB(cmn, v1)` = C-MTEB，31 任务） | 用标准 harness → 分数可比、可复现 | **全量 31 任务完成**：耗时 **173.8 min**（bs=16 / 截断 2048 / bf16 / 显存上限 70%）。总体 **0.5815**（task-type 均值）。类型均值：Classification 0.6948 / PairClassification 0.6732 / Retrieval 0.6179 / Reranking 0.5664 / Clustering 0.4925 / STS 0.4441。明细 `results/mteb_base_261008_145213.md` | 每个任务的原生分数 + 耗时 + 汇总 | ✅ |
| 1.5 | 对照组：bge-m3 / Qwen3-Embedding-0.6B | 没有对照的数字没有意义 | **bge-m3 本机实跑完成**（84.9 min，31/31 任务，同 harness/同截断/同 batch）。三方总体：**我们 0.5815 < bge-m3 0.6068 < Qwen3-Embedding-0.6B 0.6748**。**推翻了两个初判**：PAWSX 我们 0.1497 vs bge-m3 0.1566（持平 → 任务难，非模型短板）；聚类我们 0.4925 **高于** bge-m3 0.4615。真短板 = 医疗检索/reranking + AFQMC/ATEC/BQ（均落后 bge-m3 ~10 分）。7 个单项反超 bge-m3 | 同任务集三方对比表 | ✅ |
| 1.6 | 发布评测报告 | 中文圈第一个可引用数字 | **已发布**：GitHub <https://github.com/Weidows/embeddinggemma2-zh>（公开，含代码+原始分数+报告）；HF 模型卡 <https://huggingface.co/Weidows/embeddinggemma-2-zh>（公开，8 个文件已读回校验，卡片 frontmatter 解析正常，tags/license/base_model 均已生效）。卡片明确标注**暂无微调权重** | HF 模型仓库 card + GitHub README 表格 | ✅ |

## 阶段 2：中文微调

**靶子由阶段 1 的数据定死**（vs bge-m3 落后约 0.10 的两块）：

- **医疗领域**：CmedqaRetrieval、MedicalRetrieval、CMedQAv1-reranking、CMedQAv2-reranking
- **中文 STS**：AFQMC、ATEC、BQ

| # | 步骤 | 为什么 | 运行记录 | 验收成果（可独立验证） | 状态 |
|---|---|---|---|---|---|
| 2.1 | 数据可得性 + **泄漏检查** | 医疗检索的 qrels 就是评测标签，若拿它造训练对就是作弊 —— 必须先证明训练源与评测集零重叠 | **抓到一次假通过**：`cMedQA-V2.0` 的 `input` 列全为空、真问题在 `instruction` 列，按 `input` 比较得到"重叠 0"，实为在比空字符串；修正列名后**7,592 行训练问题命中 3,999 条评测查询（评测查询 100% 在训练集内）**。另测得 8,776 行训练答案命中评测语料。STS 侧：pair 级重叠 AFQMC/ATEC/PAWSX = 0，BQ 1~2、STSB 10~13、LCQMC 30（已全部剔除；LCQMC 有 43% 评测句在训练集出现过，属该数据集固有改写特性） | 重建后训练集：查询/文档重叠均为 **0**；医疗保留 120,000 对、STS 保留约 15 万对 | ⏳ 重建验证中 |
| 2.2 | 构造中文训练对 / 三元组 | 微调需要 in-domain 对比数据 | **完成**：中文 STS **~156,000** 对（AFQMC/ATEC/BQ/LCQMC/PAWSX/STSB 的 train split，取高相似度对，剔除与评测 pair 重复的 91 条）+ 医疗 **120,000** 对（cMedQA-V2.0 的 question→answer，去污染后）。前缀已按评测口径烘进文本（STS 用 `task: sentence similarity \| query: `，检索用 `task: search result \| query: ` 与 `title: none \| text: `）。产物 `data/train_pairs.jsonl`（101.9 MB）+ `data/train_pairs_sample.txt` + `train_pairs_stats.json` | 脚本重跑可复现；`check_leakage.py` 输出重叠为 0 | ✅ |
| 2.3 | LoRA 对比学习微调（bf16，禁用 fp16） | 271M 文本骨干在 16GB 卡上 LoRA 足够 | **全量完成**：bs=64 / lr 2e-4 / 1 epoch / **85.2 min**，train_loss **0.9009 → 0.2512**、eval_loss 0.2099、**峰值显存 9.85GB**（上限 11.95GB），trainable=10.92M (4.03%)。三个打包坑：① PEFT 无法包装视觉塔的 `Gemma4ClippableLinear` → 重载/校验必须带 `config_kwargs={"vision_config":None,"audio_config":None}`；② ST 用 peft 的 `inject_adapter_in_model` **原地注入** → `auto_model` 不是 PeftModel，没有 `merge_and_unload()`，且 `save_pretrained()` 只写 adapter（无 config.json/权重）；③ 单独 `mod.merge()` 后 wrapper 仍在树里（193 个）→ 必须用 `get_base_layer()` 替换。解法：自写 `merge_and_unwrap()` + `config.save_pretrained()` + `safetensors.save_file()` 手写 | 产物 `models/embeddinggemma-2-zh-lora`（271.0M / 413 张量、可独立加载、重载校验 `max\|delta\|=6.58e-04` 证明 merge 生效） | ✅ |
| 2.4 | 复跑 C-MTEB 全量并对比 | 证明微调真的有效，而非自说自话 | | 微调前后 31 任务同口径对比表（含 bge-m3 对照） | ⬜ |
| 2.5 | 发布 `Weidows/embeddinggemma-2-zh` 权重 + 卡片 | 中文第一个 v2 微调 | | HF 仓库可下载 + 第三方按 card 步骤复现出同分数 | ⬜ |

---

## 关键数据（实测）

- **编码吞吐**（5070 Ti / bf16 / 中文短文本）：
  - 合成短文本（~64 subword）：batch 64 → 1034 docs/s，batch 128 → **1201 docs/s**
  - 真实语料 + 默认 chat-template 路径：**92 docs/s**
  - 真实语料 + pin 纯 text modality：**376 docs/s**（4.1×）
- **C-MTEB 检索语料规模**（datasets-server 实测，8 个检索任务的 corpus 之和）：
  T2 118,605 / MMarco 106,813 / Du 100,001 / Covid 100,001 / Cmedqa 100,001 /
  Ecom 100,902 / Medical 100,999 / Video 100,930 → **合计 828,252 文档**。
- C-MTEB 语料文档长度：均值 332 字符、p99 1956、**max 60,975**（离群长文档 → 见坑 4）。
- 任务规模差异极大：T2Retrieval 一轮 38 分钟（默认路径），而 Ocnli 只要 34 秒。

## 已知坑

1. **`HF_ENDPOINT=https://hf-mirror.com` 会让 `hf download` 失败**
   （`Local entry not found. SSL: UNEXPECTED_EOF_WHILE_READING`）。
   可用组合：`env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 ...` 直连。
2. **mteb 2.24.0 未注册 embeddinggemma-2**（只有 `embeddinggemma-300m`）。
   → 用 `SentenceTransformerEncoderWrapper` 包本地模型，模型自带的
   `config_sentence_transformers.json` 里的 task→prompt 映射会被自动读取。
3. **`max_seq_length` 溢出**：ST 从 `max_position_embeddings`(262144) 解析出 ~1e60，等于不截断。
   必须显式 `st.max_seq_length = 2048`。
4. **长文档 OOM**：动态 padding 遇到 6 万字符的离群文档会 CUDA OOM
   → 固定 `--vram-fraction`（默认 0.7）+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
5. **依赖**：`pillow`（processor 要）+ `torchvision`（gemma4 image processor 要），
   torchvision 必须走同一个 cu128 index。
6. **整串媒体 URL 会让文本评测崩掉**（最重要的坑）：
   MMarcoRetrieval 语料第 **98275** 条是裸 YouTube 链接
   `http://www.youtube.com/ehowatHomeChannel。制作绳帘系带…`，
   ST 默认路径把它当视频去解码 → 跑 38 分钟后 `ImportError: 要解码 YouTube 视频需要先装 yt_dlp`。
   最小复现（单独输入即崩）：
   | 输入 | 结果 |
   |---|---|
   | `http://www.youtube.com/ehowatHomeChannel` | ❌ 要求 `yt_dlp` |
   | `https://example.com/clip.mp4` | ❌ libtorchcodec 报错 |
   | `https://www.baidu.com` | ✅ |
   | `看这段 https://example.com/a.mp4` | ✅（只有整串就是媒体 URL 才触发） |
   **解法**：pin 到纯 `text` modality（`src/eval_mteb.py::pin_text_modality`）。
   定位脚本：`src/find_video_trigger.py`（二分扫描语料找触发文档）。
7. **默认路径慢 4 倍**：chat-template/多模态 processor 路径 92 docs/s，
   pin 到纯 text 后 **376 docs/s**，而 200 条中文文本上两条路径的 embedding
   数值一致（`np.allclose(atol=1e-4) == True`）→ 白拿 4 倍速度、结果不变。

## 阶段 3（备选）

- 中文多模态评测（MIEB / MAEB 的中文子集）——需要另配数据，暂缓。

## 关键决策记录

- **只做文本评测起步**：视觉/音频评测需要另配数据（MIEB/MAEB），先出文本中文数字，
  多模态中文评测（阶段 3 备选）另开。
- **用 mteb 而非手搓指标**：数字要能对标榜单；mteb 已验证可用（见坑 2 的绕法）。
- **一律 pin 纯 text modality**：既绕开坑 6 的崩溃，又白拿 4 倍速度，且 embedding 不变。
- **禁用 fp16**：任何脚本里 dtype 一律 bf16（GPU）/ fp32（CPU）。
- **显存纪律**：`--vram-fraction 0.7` 硬上限 + `expandable_segments:True`，宁可自己 OOM 也不拖垮整机。
