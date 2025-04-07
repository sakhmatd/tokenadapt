### Model Information

| Feature          | Value                             |
| :--------------- | :-------------------------------- |
| Model            | `tinycompany/Llamaify-T0.6-3B`    |
| Base Model       | `meta-llama/Llama-3.2-3B`         |
| New-Tokenizer    | `tinycompany/Adi-Bun-128K`        |
| Unique Tokens    | `108K`                            |
| Common Tokens    | `20k`                             |

---

Note:-> The tokenizer used is trained while keeping `supertokens` in mind. (Similar to SuperBPE, Although training algorithm is different)

* `supertokens` :-> Multi-word token like 
  ```python 
  [
    'is the',
    'ed to', 
    # In other tokenizer ; token is either a single word 
    # Or a subword, or a word with whitespace prefix.
    # But this allows tokenizer to learn multi-word representation 
    'I am', etc..
  ]
  ```


### Configuration Details

**1. Local Heuristic Only**
*   Model Used: `tinycompany/Llamified-128k-1` (Implied from perplexity results section)
*   Config:
    ```json
    {
      "temperature": 0.8,
      "top_k": 3,
      "weight": 0.0,
      "embedding_model": "nomic-ai/nomic-embed-text-v2-moe"
    }
    ```

**2. Local + Global Heuristic**
*   Model Used: `tinycompany/Llamaify-T0.6-3B`
*   Config:
    ```json
    {
      "temperature": 0.6,
      "top_k": 3,
      "weight": 0.2,
      "embedding_model": "nomic-ai/nomic-embed-text-v2-moe"
    }
    ```

---

### Perplexity Comparison on `tinycompany/ppl` Dataset

| Subset         | Metric | Local Heuristic Only (`Llamified-128k-1`) | Local + Global Heuristic (`Llamaify-T0.6-3B`) |
| :------------- | :----- | :---------------------------------------- | :------------------------------------------ |
| **Overall**    | Mean   | 27426.06                                  | 11550.15                                    |
|                | Median | 1874.22                                   | 1133.02                                     |
|                | Std    | 56909.00                                  | 26418.87                                    |
|                | Range  | 42.21 to 556315.44                        | 50.18 to 207898.12                          |
|---|---|---|---|---|
| **English**    | Mean   | 2199.43                                   | 1128.16                                     |
|                | Median | 1838.06                                   | 974.13                                      |
|                | Std    | 1464.65                                   | 605.18                                      |
|                | Range  | 268.46 to 9116.49                         | 281.78 to 3993.66                           |
|---|---|---|---|---|
| **Code**       | Mean   | 523.17                                    | 561.30                                      |
|                | Median | 344.05                                    | 474.04                                      |
|                | Std    | 858.90                                    | 411.27                                      |
|                | Range  | 66.19 to 8213.59                          | 59.04 to 2479.50                            |
|---|---|---|---|---|
| **Math**       | Mean   | 496.80                                    | 547.70                                      |
|                | Median | 262.44                                    | 400.29                                      |
|                | Std    | 688.80                                    | 538.76                                      |
|                | Range  | 42.21 to 3553.45                          | 50.18 to 3130.06                            |
|---|---|---|---|---|
| **Hinglish**   | Mean   | 118091.26                                 | 52809.09                                    |
|                | Median | 93046.19                                  | 41997.04                                    |
|                | Std    | 90301.09                                  | 44890.80                                    |
|                | Range  | 7962.57 to 556315.44                      | 2936.87 to 207898.12                        |
|---|---|---|---|---|
| **Hindi**      | Mean   | 41463.67                                  | 13120.57                                    |
|                | Median | 43056.03                                  | 12674.33                                    |
|                | Std    | 16808.59                                  | 5683.83                                     |
|                | Range  | 1420.88 to 85572.61                       | 805.13 to 24960.59                          |


* P.S:-> The perplexity values of ReTok or Just averaging subword weighting was around 28k. 