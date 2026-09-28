
> *Note: This folder contains the text of the Master's thesis in chess AI. This and all the subsequent documents are meant to be transposed to LaTex in pdf form.*

[[related-work]]

---

## Notes for the AI agents

- professional, scientific tone
- mind the sources!
- no lists or bullet points unless absolutely necessary
- watch out for duplicate information and redundancy - do not state the same exact thing in two different chapters
- not too many "we", "our", and "us" - sometimes you can just refer to the algorithm/method instead
- not too many em dashes (option: en dashes among spaces)
- watch out for formula glitches ("BB" instead of "$B$", "\(a+b\)" instead of "$a+b$")
- check for inconsistencies: chapters in the wrong place, wrong summary title
- check "thesis-writer" skill if available
- ~~strikethrough~~ text is not meant to be exported to the LaTex file
- <span style="color: #808080;">[Each paragraph must have a clear point]</span>, marked in brackets in gray at the beginning like this
- **bald** and *italic* are meant to be transposed to the LaTex, unless stated otherwise

---

## Hyperparameters

> *Note for the AI: refer to this file for every hyperparameter and variable. When converting markdown to LaTex, replace every square-bracket placeholder (e.g. `[dataset_size]`, `[W]`) with the **current value** from the table below. Do not leave placeholders in the LaTex.*

| Placeholder             | Meaning                                                       | Current value                                       |
| :---------------------- | :------------------------------------------------------------ | :-------------------------------------------------- |
| `[dataset_size]`        | Dataset size (FEN positions, with duplicates)                 | 105M                                                |
| `[dataset_size_unique]` | Dataset size (unique FEN positions)                           | 104M                                                |
| `[test_fraction]`       | Percentage of the dataset held out as test set                | 1%                                                  |
| `[batch_size]`          | Training batch size                                           | 10k                                                 |
| `[lr_start]`            | Initial learning rate                                         | $10^{-2}$                                           |
| `[lr_end]`              | Final (decayed) learning rate                                 | $10^{-3}$                                           |
| `[d_in]`                | Input / sparse-feature dimension                              | 844                                                 |
| `[W]`                   | L1 accumulator width **per perspective** (dual NNUE: `[W]`×2) | 128                                                 |
| `[H]`                   | L2 hidden width                                               | 256                                                 |
| `[model_size]`          | total number of model parameters                              | 70k                                                 |
| `[keep_prob]`           | Probability of keeping any sampled position in the dataset    | 5%                                                  |
| `[n_params]`            | Total trainable parameters of the target architecture         | *derive from `[d_in]`, `[W]`, `[H]` at export time* |
| `[P_head]`              | Flattened head-parameter dimension (L2 + output)              | *derive from `[W]`, `[H]` at export time*           |

Target model architecture: $d_{\text{in}}=[d_in]$, dual NNUE, L1 accumulator layer $[W]\times 2$ neurons, L2 $[H]$ neurons, output WDL for the current player.

---
