# MTEB run — models/embeddinggemma-2-zh-lora (finetuned)

- benchmark: `MTEB(cmn, v1)` (31 tasks)
- device: cuda / torch.bfloat16 / batch_size=16
- wall time: 137.8 min
- mteb: 2.24.0

| task | type | split | main metric | score |
|---|---|---|---|---|
| T2Retrieval | Retrieval | dev | ndcg_at_10 | 0.7292 |
| MMarcoRetrieval | Retrieval | dev | ndcg_at_10 | 0.6997 |
| DuRetrieval | Retrieval | dev | ndcg_at_10 | 0.7243 |
| CovidRetrieval | Retrieval | dev | ndcg_at_10 | 0.6433 |
| CmedqaRetrieval | Retrieval | dev | ndcg_at_10 | 0.3361 |
| EcomRetrieval | Retrieval | dev | ndcg_at_10 | 0.5532 |
| MedicalRetrieval | Retrieval | dev | ndcg_at_10 | 0.4726 |
| VideoRetrieval | Retrieval | dev | ndcg_at_10 | 0.5557 |
| T2Reranking | Reranking | dev | map_at_1000 | 0.6587 |
| MMarcoReranking | Reranking | dev | map_at_1000 | 0.2088 |
| CMedQAv1-reranking | Reranking | test | map_at_1000 | 0.8003 |
| CMedQAv2-reranking | Reranking | test | map_at_1000 | 0.8200 |
| Ocnli | PairClassification | validation | max_accuracy | 0.5826 |
| Cmnli | PairClassification | validation | max_accuracy | 0.6672 |
| CLSClusteringS2S | Clustering | test | v_measure | 0.3767 |
| CLSClusteringP2P | Clustering | test | v_measure | 0.4068 |
| ThuNewsClusteringS2S | Clustering | test | v_measure | 0.5628 |
| ThuNewsClusteringP2P | Clustering | test | v_measure | 0.6383 |
| LCQMC | STS | test | cosine_spearman | 0.7251 |
| PAWSX | STS | test | cosine_spearman | 0.1423 |
| AFQMC | STS | validation | cosine_spearman | 0.3685 |
| QBQTC | STS | test | cosine_spearman | 0.3006 |
| TNews | Classification | validation | accuracy | 0.4903 |
| IFlyTek | Classification | validation | accuracy | 0.4948 |
| Waimai | Classification | test | accuracy | 0.8481 |
| OnlineShopping | Classification | test | accuracy | 0.8811 |
| JDReview | Classification | test | accuracy | 0.7587 |
| MultilingualSentiment | Classification | validation | accuracy | 0.6851 |
| MultilingualSentiment | Classification | test | accuracy | 0.6886 |
| ATEC | STS | test | cosine_spearman | 0.4344 |
| BQ | STS | test | cosine_spearman | 0.6396 |
| STSB | STS | test | cosine_spearman | 0.7866 |
