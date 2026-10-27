# Harmful-prompt benchmarks

Used only to evaluate harmlessness (each answer is judged safe / unsafe).

| File | Source | Size |
|---|---|---|
| `jailbreakbench.csv` | JailbreakBench behaviors, Chao et al., NeurIPS 2024 Datasets & Benchmarks ([repo](https://github.com/JailbreakBench/jailbreakbench)) | 100 |
| `maliciousinstruct.csv` | MaliciousInstruct, Huang et al., ICLR 2024 ([repo](https://github.com/Princeton-SysML/Jailbreak_LLM)) | 100 |

Each file has a single column, `input`. Please cite the original works when using these data.
The prompts are harmful by construction.
