# MTEB run — models/embeddinggemma-2-zh-lora (finetuned_weak)

- benchmark: `custom` (8 tasks)
- device: cuda / torch.bfloat16 / batch_size=16
- wall time: 29.3 min
- mteb: 2.24.0

| task | type | split | main metric | score |
|---|---|---|---|---|
| AFQMC | STS | validation | cosine_spearman | 0.3685 |
| ATEC | STS | validation | cosine_spearman | 0.4398 |
| ATEC | STS | test | cosine_spearman | 0.4344 |
| BQ | STS | validation | cosine_spearman | 0.6745 |
| BQ | STS | test | cosine_spearman | 0.6396 |
| LCQMC | STS | test | cosine_spearman | 0.7251 |
| MedicalRetrieval | Retrieval | dev | ndcg_at_10 | 0.4726 |
| CmedqaRetrieval | Retrieval | dev | ndcg_at_10 | 0.3361 |
| CMedQAv1-reranking | Reranking | test | map_at_1000 | 0.8003 |
| CMedQAv2-reranking | Reranking | test | map_at_1000 | 0.8200 |
