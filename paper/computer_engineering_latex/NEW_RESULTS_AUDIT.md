# 新实验结果审计与论文更新说明

## 数据目录

本次检查的数据根目录为：

`C:\Users\杨昊\Desktop\实验数据\owl\owl`

审计仅纳入状态为 `success`、包含完整评价结果且实验口径能够从结果文件确认的记录。论文数值由 CSV 或 `result.json` 直接读取，没有修改或推断原始实验数据。

## 已写入论文的结果

### 1. 扩展 zero-shot 搜索

`zero_shot_wsqrtg_v9_remaining/zero_shot_topk_summary.csv` 共包含640条成功记录。新一轮搜索使4个主表单元格提高：

| 模型 | 稀疏率 | 原结果 / % | 新结果 / % | 变化 / 百分点 |
|---|---:|---:|---:|---:|
| LLaMA-1-7B | 50% | 55.430 | 55.544 | +0.114 |
| LLaMA-1-7B | 60% | 50.787 | 51.067 | +0.280 |
| LLaMA-2-7B | 50% | 55.671 | 55.677 | +0.006 |
| LLaMA-2-7B | 60% | 50.440 | 50.781 | +0.341 |

LLaMA-1-13B、LLaMA-1-30B、LLaMA-2-13B 和全部70%稀疏率结果未出现更高的逐任务上界。论文表2及图3已更新。更新后，LLaMA-1-7B在60%稀疏率下的51.067%超过 SparseGPT 的51.001%，因此该列的最优和次优标记也已交换。

### 2. Vicuna-7B 参数实验

`owl_v9_wsqrtg_param_vicuna_7b_l20_2gpu/summary_all.csv` 包含360条成功记录，每档稀疏率120条。最低 WikiText-2 PPL 为：

| 模型 | 方法 | 50% | 60% | 70% |
|---|---|---:|---:|---:|
| Vicuna-7B | GS-OWL | 7.9343 | 10.4193 | 32.9239 |

相对于当前论文数据中的 SparseGPT，三档 PPL 分别降低4.1%、11.2%和2.2%。该结果已加入表6，并在实验设置、摘要和讨论中将扩展模型范围更新为 Mistral-7B 与 Vicuna-7B。

### 3. C4 跨语料验证

`c4_validation_gs_owl/summary_c4.csv` 包含5个 LLaMA 模型、3档稀疏率，共15条成功记录。所有运行均使用 WikiText-2 主实验确定的固定配置，评价集为 C4 validation，不在 C4 上重新选择配置。论文新增表7：

| 模型 | 50% | 60% | 70% |
|---|---:|---:|---:|
| LLaMA-1-7B | 9.1540 | 11.9329 | 28.4379 |
| LLaMA-1-13B | 8.0135 | 9.7370 | 16.3837 |
| LLaMA-1-30B | 7.1012 | 8.5464 | 12.9034 |
| LLaMA-2-7B | 8.7936 | 11.7482 | 37.5006 |
| LLaMA-2-13B | 7.7821 | 9.6536 | 18.4997 |

该表只支持“固定配置在评价语料变化后仍呈现可解释的单调退化趋势”，不用于在缺少同协议基线时宣称方法优越性。

## 暂未写入论文的结果

### 1. OSLA 内部组件与边界实验

`v9_component_ablation/summary_all.csv` 的20条记录和 `osla_mechanism_ablation/summary_all.csv` 的10条记录均成功，但结果不能支持“每增加一个内部步骤都会单调改善 PPL”的叙述。例如：

- LLaMA-1-7B、70%：anchored 为20.7138，full 为20.8709；
- LLaMA-2-7B、70%：count 为25.2871，full 为25.5254；
- 部分左移边界设置略优于当前固定边界；
- protected projection 并未在所有设置中优于 uniform projection。

因此这些结果不适合被选择性整理为“逐项均有效”的消融表。本稿继续采用表3中的整体 OSLA 与原始 OWL 匹配比较，只论证完整层分配策略相对原始 OWL 的总体效果；关于严重度、核心区、尾部限制和预算投影的独立贡献仍作为审稿风险保留。

### 2. OPT-1.3B 与 Mistral/Qwen zero-shot

- OPT-1.3B 已有完整 PPL 和 zero-shot 结果，但当前稿件没有同协议的标准基线表，直接加入只会形成 GS-OWL 单方法结果，暂不纳入正文主表。
- Mistral-7B 的 Top10 单配置 zero-shot 平均准确率为59.196%、53.291%和41.900%，但缺少当前论文统一口径下的 Wanda、SparseGPT、RIA 和 Pruner-Zero 对照，暂不加入表2。
- Qwen2.5-7B 的 PPL 为异常高值，且其 zero-shot 准确率接近随机水平，继续保持不纳入论文结论。

## 本次修改文件

- `main.tex`
- `data/ppl_main_results.csv`
- `data/zero_shot_main_results.csv`
- `data/c4_validation_results.csv`
- `figures/zero_shot_trends_color.pdf`
- `figures/zero_shot_trends_bw.pdf`
- `README.md`

