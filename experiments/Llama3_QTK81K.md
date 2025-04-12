### Model Information

| Feature          | Value                             |
| :--------------- | :-------------------------------- |
| Model            | `tinycompany/Llamaify-inf-nml-3B` |
| Model            | `tinycompany/Llamaify-inf-nmlg-3B`|
| Base Model       | `meta-llama/Llama-3.2-3B`         |
| New-Tokenizer    | `fhai50032/QTK-81K`               |
| Unique Tokens    | `48K`                             |
| Common Tokens    | `33K`                             |

---

### Configuration Details

**1. Local Heuristic Only**
*   Model Used: `tinycompany/Llamaify-inf-nml-3B` (Implied from perplexity results section)
*   Config:
    ```json
    {
      "temperature": 0.6,
      "top_k": 3,
      "weight": 0.0,
      "embedding_model": "nomic-ai/nomic-embed-text-v2-moe"
    }
    ```

**2. Local + Global Heuristic**
*   Model Used: `tinycompany/Llamaify-inf-nmlg-3B`
*   Config:
    ```json
    {
      "temperature": 0.6,
      "top_k": 3,
      "weight": 0.3,
      "embedding_model": "nomic-ai/nomic-embed-text-v2-moe"
    }
    ```

---

### Perplexity Comparison on `tinycompany/ppl` Dataset
| Subset       | Metric   | Local Heuristic Only (`Llamaify-inf-nml-3B`) | Local + Global Heuristic (`Llamaify-inf-nmlg-3B`) |
|-------------|----------|---------------------------------------------:|--------------------------------------------------:|
| **Overall** | Mean     | 1607.4772                                  | 1201.3640                                       |
|             | Median   | 7.5000                                     | 7.3750                                          |
|             | Std      | 3345.3394                                  | 2577.7769                                       |
|             | Range    | 1.8516 to 20736.0000                       | 1.8359 to 23424.0000                            |
| **English** | Mean     | 3.24446                                    | 3.4271                                          |
|             | Median   | 3.1562                                     | 3.1250                                          |
|             | Std      | 1.1068                                     | 1.1082                                          |
|             | Range    | 1.8516 to 9.1875                           | 1.8359 to 8.9375                                |
| **Code**    | Mean     | 8.0834                                     | 7.8006                                          |
|             | Median   | 5.9375                                     | 5.7969                                          |
|             | Std      | 5.9951                                     | 5.5505                                          |
|             | Range    | 2.0625 to 35.2500                          | 2.0312 to 32.5000                               |
| **Math**    | Mean     | 12.5325                                    | 13.3822                                         |
|             | Median   | 8.6875                                     | 8.6875                                          |
|             | Std      | 30.6120                                    | 39.4924                                         |
|             | Range    | 2.8906 to 314.0000                         | 2.8750 to 404.0000                              |
| **Hinglish**| Mean     | 1839.3637                                  | 2055.9025                                       |
|             | Median   | 1168.0000                                  | 1344.0000                                       |
|             | Std      | 2282.9273                                  | 2924.8667                                       |
|             | Range    | 30.6250 to 12544.0000                      | 29.2500 to 20736.0000                           |
| **Hindi**   | Mean     | 7856.5253                                  | 5175.9697                                       |
|             | Median   | 7616.0000                                  | 4608.0000                                       |
|             | Std      | 3616.4179                                  | 3052.2491                                       |
|             | Range    | 296.0000 to 20736.0000                     | 158.0000 to 23424.0000                          |


* P.S:-> The perplexity values of ReTok or Just averaging subword weighting was around 28k. 
