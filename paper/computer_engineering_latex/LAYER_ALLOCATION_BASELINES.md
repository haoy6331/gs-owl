# Layer-allocation baseline extraction

The values below are transcribed from the cited papers. They are useful as
published references, but they should not be described as locally reproduced
results because tokenizer, dataset split, pruning implementation, and library
versions may differ from this repository.

## 70% unstructured sparsity with Wanda

| Method | LLaMA-1-7B | LLaMA-1-13B | LLaMA-1-30B | LLaMA-2-7B | LLaMA-2-13B | Primary location |
|---|---:|---:|---:|---:|---:|---|
| OWL | 24.46 | 16.23 | 10.77 | 30.58 | 20.65 | DLP paper, Table 3 (re-reported OWL baseline) |
| AlphaPruning | 23.86 | 14.21 | 9.68 | 28.87 | 14.16 | AlphaPruning paper, Table 2 |
| DSA | 22.60 | -- | -- | -- | -- | DSA paper, Table 5 |
| DLP | 20.46 | 13.65 | 9.93 | 22.79 | 16.19 | DLP paper, Table 3 |
| ATP | 20.16 | -- | -- | 22.16 | -- | ATP paper, Table 3 |
| GS-OWL (local) | 20.8710 | 11.9627 | 9.0048 | 25.5248 | 12.2744 | Current manuscript, Table 1 |

## Protocol notes

- OWL originally uses 128 C4 calibration sequences of length 2,048 and reports
  WikiText validation perplexity. Its original Table 3 reports 24.55, 17.17,
  and 10.75 for LLaMA-1-7B/13B/30B with Wanda at 70% sparsity.
- AlphaPruning reports WikiText validation perplexity and combines each layer
  allocator with a specified within-layer pruner. The row used in the manuscript
  is the Wanda row.
- DSA searches an allocation function on a validation set. Its Table 5 reports
  only LLaMA-1-7B for this particular perplexity comparison.
- DLP uses 128 C4 calibration samples of length 2,048. Its Table 3 supplies a
  five-model Wanda comparison under one paper implementation.
- ATP determines a monotonic arithmetic-progression allocation and searches its
  step parameter on WikiText-2. Its Table 3 publishes the two 7B Wanda results.

## Manuscript use

Use these values only in rows marked as published references. A definitive
same-protocol comparison still requires running the authors' allocation code
with this repository's tokenizer, calibration samples, sparsity definition,
and WikiText2 evaluator.
