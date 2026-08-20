=== Qwen3.5-2B-UD-Q4_K_XL.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 2B Q4_K - Medium        |   1.24 GiB |     1.88 B | CPU        |       6 |           pp512 |         91.13 ± 2.24 |
| qwen35 2B Q4_K - Medium        |   1.24 GiB |     1.88 B | CPU        |       6 |           tg128 |         14.89 ± 0.15 |

build: a3b1eff (1)
=== LFM2.5-2.6B-QAD-Q4_0.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| lfm2 2.6B Q4_0                 |   1.48 GiB |     2.70 B | CPU        |       6 |           pp512 |         79.32 ± 0.90 |
| lfm2 2.6B Q4_0                 |   1.48 GiB |     2.70 B | CPU        |       6 |           tg128 |         13.41 ± 0.04 |

build: a3b1eff (1)
=== LFM2-2.6B-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| lfm2 2.6B Q4_K - Medium        |   1.45 GiB |     2.57 B | CPU        |       6 |           pp512 |         78.43 ± 1.92 |
| lfm2 2.6B Q4_K - Medium        |   1.45 GiB |     2.57 B | CPU        |       6 |           tg128 |         13.69 ± 0.07 |

build: a3b1eff (1)
=== Qwen3-1.7B-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | CPU        |       6 |           pp512 |        151.81 ± 3.89 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | CPU        |       6 |           tg128 |         20.64 ± 0.04 |

build: a3b1eff (1)
=== granite-4.1-3b-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| granite 3B Q4_K - Medium       |   1.95 GiB |     3.40 B | CPU        |       6 |           pp512 |         67.62 ± 1.60 |
| granite 3B Q4_K - Medium       |   1.95 GiB |     3.40 B | CPU        |       6 |           tg128 |         11.08 ± 0.03 |

build: a3b1eff (1)
=== granite-4.0-h-micro-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| granitehybrid 3B Q4_K - Medium |   1.81 GiB |     3.19 B | CPU        |       6 |           pp512 |         68.82 ± 0.24 |
| granitehybrid 3B Q4_K - Medium |   1.81 GiB |     3.19 B | CPU        |       6 |           tg128 |         10.64 ± 0.01 |

build: a3b1eff (1)
=== SmolLM3-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| smollm3 3B Q4_K - Medium       |   1.78 GiB |     3.08 B | CPU        |       6 |           pp512 |         76.07 ± 0.60 |
| smollm3 3B Q4_K - Medium       |   1.78 GiB |     3.08 B | CPU        |       6 |           tg128 |         12.05 ± 0.12 |

build: a3b1eff (1)
=== Qwen3.5-4B-Q3_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q3_K - Medium        |   2.13 GiB |     4.21 B | CPU        |       6 |           pp512 |         39.20 ± 0.02 |
| qwen35 4B Q3_K - Medium        |   2.13 GiB |     4.21 B | CPU        |       6 |           tg128 |          8.82 ± 0.01 |

build: a3b1eff (1)
=== Qwen3-4B-Instruct-2507-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       6 |           pp512 |         56.07 ± 2.30 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       6 |           tg128 |          9.31 ± 0.02 |

build: a3b1eff (1)
=== gemma-3-4b-it-qat-Q4_0.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gemma3 4B Q4_0                 |   2.35 GiB |     3.88 B | CPU        |       6 |           pp512 |         64.27 ± 0.41 |
| gemma3 4B Q4_0                 |   2.35 GiB |     3.88 B | CPU        |       6 |           tg128 |          9.07 ± 0.01 |

build: a3b1eff (1)
BENCH-FINISHED
=== Phi-4-mini-instruct-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CPU        |       6 |           pp512 |         52.50 ± 0.56 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CPU        |       6 |           tg128 |          8.83 ± 0.03 |

build: a3b1eff (1)
