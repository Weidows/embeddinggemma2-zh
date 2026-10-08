# C-MTEB (MTEB(cmn, v1)) — embeddinggemma-2 vs bge-m3 vs Qwen3-Embedding-0.6B

Every column uses **the same harness** (mteb 2.24.0, `MTEB(cmn, v1)`, 31 tasks).
`local run` = executed on this machine (RTX 5070 Ti, bf16, truncate 2048, batch 16);
other columns are the published `mteb/results` entries for the same task names.

| column | tasks covered | source |
|---|---|---|
| embeddinggemma-2 (local run) | 31/31 | local run |
| bge-m3 (local run) | 31/31 | local run |
| BAAI__bge-m3 | 5/31 | mteb/results (public) |
| google__embeddinggemma-300m | 3/31 | mteb/results (public) |
| Qwen__Qwen3-Embedding-0.6B | 31/31 | mteb/results (public) |

## Full 31-task table

| task | type | embeddinggemma-2 (local run) | bge-m3 (local run) | Qwen__Qwen3-Embedding-0.6B |
|---|---|---|---|---|
| T2Retrieval | Retrieval | 0.7618 | 0.8149 | 0.8368 |
| MMarcoRetrieval | Retrieval | 0.7825 | 0.7726 | 0.7985 |
| DuRetrieval | Retrieval | 0.7637 | 0.8399 | 0.8410 |
| CovidRetrieval | Retrieval | 0.7721 | 0.7751 | 0.8476 |
| CmedqaRetrieval | Retrieval | 0.2451 | 0.3374 | 0.4109 |
| EcomRetrieval | Retrieval | 0.6040 | 0.5845 | 0.6432 |
| MedicalRetrieval | Retrieval | 0.4335 | 0.5427 | 0.5604 |
| VideoRetrieval | Retrieval | 0.5808 | 0.5699 | 0.7436 |
| T2Reranking | Reranking | 0.6613 | 0.6688 | 0.6715 |
| MMarcoReranking | Reranking | 0.2511 | 0.2754 | 0.2175 |
| CMedQAv1-reranking | Reranking | 0.6650 | 0.7769 | 0.8006 |
| CMedQAv2-reranking | Reranking | 0.6882 | 0.7909 | 0.8135 |
| Ocnli | PairClassification | 0.6297 | 0.6508 | 0.7298 |
| Cmnli | PairClassification | 0.7167 | 0.7183 | 0.7986 |
| CLSClusteringS2S | Clustering | 0.3743 | 0.3827 | 0.5828 |
| CLSClusteringP2P | Clustering | 0.3844 | 0.3899 | 0.6062 |
| ThuNewsClusteringS2S | Clustering | 0.5803 | 0.5107 | 0.7457 |
| ThuNewsClusteringP2P | Clustering | 0.6311 | 0.5627 | 0.8150 |
| LCQMC | STS | 0.7253 | 0.7618 | 0.7646 |
| PAWSX | STS | 0.1497 | 0.1566 | 0.2763 |
| AFQMC | STS | 0.2783 | 0.4000 | 0.4323 |
| QBQTC | STS | 0.3372 | 0.3331 | 0.3707 |
| ATEC | STS | 0.3447 | 0.4647 | 0.4858 |
| BQ | STS | 0.4731 | 0.5739 | 0.6466 |
| STSB | STS | 0.8002 | 0.8060 | 0.8462 |
| TNews | Classification | 0.4673 | 0.4961 | 0.5296 |
| IFlyTek | Classification | 0.4302 | 0.4768 | 0.5167 |
| Waimai | Classification | 0.8499 | 0.8665 | 0.8129 |
| OnlineShopping | Classification | 0.8991 | 0.9206 | 0.8360 |
| JDReview | Classification | 0.7964 | 0.8008 | 0.8041 |
| MultilingualSentiment | Classification | 0.7255 | 0.7168 | 0.7897 |

## Aggregate

| | embeddinggemma-2 (local run) | bge-m3 (local run) | Qwen__Qwen3-Embedding-0.6B |
|---|---|---|---|
| Retrieval | 0.6179 | 0.6546 | 0.7103 |
| Reranking | 0.5664 | 0.6280 | 0.6258 |
| PairClassification | 0.6732 | 0.6846 | 0.7642 |
| Clustering | 0.4925 | 0.4615 | 0.6874 |
| STS | 0.4441 | 0.4994 | 0.5461 |
| Classification | 0.6948 | 0.7129 | 0.7149 |
| **Overall (mean of type means)** | **0.5815** | **0.6068** | **0.6748** |

## Delta vs `embeddinggemma-2 (local run)` (positive = ours better)

| task | type | bge-m3 (local run) | Qwen__Qwen3-Embedding-0.6B |
|---|---|---|---|
| T2Retrieval | Retrieval | -0.0532 | -0.0750 |
| MMarcoRetrieval | Retrieval | +0.0099 | -0.0160 |
| DuRetrieval | Retrieval | -0.0762 | -0.0773 |
| CovidRetrieval | Retrieval | -0.0029 | -0.0755 |
| CmedqaRetrieval | Retrieval | -0.0923 | -0.1658 |
| EcomRetrieval | Retrieval | +0.0196 | -0.0391 |
| MedicalRetrieval | Retrieval | -0.1092 | -0.1269 |
| VideoRetrieval | Retrieval | +0.0108 | -0.1629 |
| T2Reranking | Reranking | -0.0075 | -0.0102 |
| MMarcoReranking | Reranking | -0.0243 | +0.0336 |
| CMedQAv1-reranking | Reranking | -0.1120 | -0.1356 |
| CMedQAv2-reranking | Reranking | -0.1026 | -0.1253 |
| Ocnli | PairClassification | -0.0211 | -0.1002 |
| Cmnli | PairClassification | -0.0017 | -0.0819 |
| CLSClusteringS2S | Clustering | -0.0084 | -0.2086 |
| CLSClusteringP2P | Clustering | -0.0055 | -0.2218 |
| ThuNewsClusteringS2S | Clustering | +0.0695 | -0.1654 |
| ThuNewsClusteringP2P | Clustering | +0.0684 | -0.1839 |
| LCQMC | STS | -0.0365 | -0.0393 |
| PAWSX | STS | -0.0069 | -0.1266 |
| AFQMC | STS | -0.1218 | -0.1540 |
| QBQTC | STS | +0.0042 | -0.0335 |
| ATEC | STS | -0.1200 | -0.1411 |
| BQ | STS | -0.1008 | -0.1735 |
| STSB | STS | -0.0059 | -0.0460 |
| TNews | Classification | -0.0288 | -0.0623 |
| IFlyTek | Classification | -0.0466 | -0.0865 |
| Waimai | Classification | -0.0166 | +0.0370 |
| OnlineShopping | Classification | -0.0215 | +0.0631 |
| JDReview | Classification | -0.0043 | -0.0077 |
| MultilingualSentiment | Classification | +0.0088 | -0.0642 |
| **Overall** | | **-0.0254** | **-0.0933** |

## Sparse public entries — 3-task intersection only

These models have published results for only part of the benchmark, so they
are compared on the intersection, not on their own averages.

| task | type | embeddinggemma-2 (local run) | BAAI__bge-m3 | google__embeddinggemma-300m |
|---|---|---|---|---|
| CovidRetrieval | Retrieval | 0.7721 | 0.7751 | 0.7893 |
| T2Reranking | Reranking | 0.6613 | 0.6683 | 0.6754 |
| STSB | STS | 0.8002 | 0.8190 | 0.8164 |
| **Overall (on intersection)** | | **0.7445** | **0.7541** | **0.7604** |
