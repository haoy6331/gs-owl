# 《计算机工程》LaTeX 稿件

## 文件说明

- `main.tex`：论文主文件。
- `references.tex`：顺序编码制参考文献。
- `make_figures.py`：生成方法总览、PPL、zero-shot 和层间分配图的彩色版与黑白版。
- `figures/`：PDF 矢量图和 PNG 预览图；正文默认调用 `*_color.pdf`。
- `data/`：主结果、zero-shot、C4 跨语料验证、严格消融和层间分布数据。
- `submission_info.tex`：作者联系方式和投稿附加信息，不参与正文编译。
- `build/`：本地编译产物。

## 编译

推荐使用 XeLaTeX。Overleaf 中将编译器设置为 `XeLaTeX`，然后编译 `main.tex`。

本地使用 Tectonic：

```bash
tectonic -X compile main.tex --outdir build
```

重新生成全部彩色图和黑白图：

```bash
python make_figures.py
```

正文中的 `\figsuffix` 默认为 `color`。如需检查黑白打印效果，把
`main.tex` 中的

```tex
\newcommand{\figsuffix}{color}
```

改为：

```tex
\newcommand{\figsuffix}{bw}
```

## Overleaf 上传清单

最小可编译材料：

1. `main.tex`
2. `references.tex`
3. `figures/method_overview_final_color.pdf`
4. `figures/main_ppl_trends_color.pdf`
5. `figures/zero_shot_trends_color.pdf`
6. `figures/layer_allocation_color.pdf`
7. `figures/pllama_domain_mcq_color.pdf`

若需要随时切换为黑白稿，再上传对应的 4 个 `*_bw.pdf`。`make_figures.py` 和
`data/` 不参与 Overleaf 编译，可作为复现材料单独保存。

## 投稿前必须填写

1. 中英文作者姓名、作者顺序和单位。
2. 基金项目名称与编号。
3. 通信作者长期有效邮箱。
4. 前 3 位作者的电话、手机和邮箱。
5. 第一作者的 CCF 会员级别及会员号；不是会员则删除。
6. 保密审查证明和著作权转让协议。

## 数据口径提醒

- Dense FP16 为公开文献参考值，表中已用 `dagger` 标注。
- 论文中的正式方法名为 GS-OWL，层级分配模块为 OSLA，连接显著性模块为 GWS；代码中的历史 `v9`
  标识仅用于实验目录兼容，不再作为论文术语。
- 当前 zero-shot 表按任务从候选掩码中取最高准确率，正文已在表下注明该口径；单一掩码结果仍是更严格的投稿版本。
- GWS 的 13.9% 时间增量对应复用离线梯度后的重复剪枝阶段。首次梯度生成是一次性预处理成本，可跨稀疏率和配置摊销。
- 主 PPL 表中的 `\ddagger` 行为 OWL、AlphaPruning、DSA、DLP 和 ATP 的论文公开值，并非本地同协议复现结果。
- C4 跨语料副实验使用 `scripts/run_c4_auxiliary.py`；固定主实验配置后运行，不在 C4 上重新选择配置。
- Qwen2.5-7B 当前 PPL 异常，未纳入正文结论。
- Vicuna-7B 的360组参数实验已完成，正文表6纳入其 WikiText-2 PPL 及 Wanda、SparseGPT 对照。
- PLLaMA-7B/13B 领域实验使用 MoBiPlant 的565道专家选择题；每个稀疏率报告20个候选掩码中的最高准确率，
  正文仅与对应 Dense 模型比较。
- C4 表使用 WikiText-2 主实验确定的固定配置，不在 C4 验证结果上重新选择配置。
- 非结构化稀疏不等同于通用 GPU 上的线性推理加速。

## 期刊格式说明

《计算机工程》官网当前要求投稿系统上传 Word 文件，并按黑白印刷检查图片。该项目用于 LaTeX 写作、
版本管理和 PDF 定稿；正式投稿前应向编辑部确认能否直接提交 PDF，或将最终稿转换为 Word。
