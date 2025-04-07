
### Model & Tokenizer Information

| Feature                     | Value                             |
| :-------------------------- | :-------------------------------- |
| Base Model                  | `Qwen/Qwen2.5-3B`                 |
| New Tokenizer               | `fhai50032/QTK-81K`               |
| Shared Tokens               | `46570`                           |
| Unique Tokens to Initialize | `35350`                           |
| Embedding Model             | `nomic-ai/nomic-embed-text-v2-moe`|

--- 



### Configuration Details

| Configuration        | Model Name                       | Weight | Temperature | Top_k | Notes                                     |
| :------------------- | :------------------------------- | :----- | :---------- | :---- | :---------------------------------------- |
| No Global (Local)    | `tinycompany/Qwentify-No-Global` | 0.0    | 0.3         | 3     | Uses only local heuristic     |
| Global + Local       | `tinycompany/Qwentify-Global_local`| 0.3    | 0.3         | 3     | Combines local and global (weight=0.3)    |
| ReTok                | `tinycompany/Qwentify-ReTok`     | *N/A*  | *N/A*       | *N/A* | Configuration parameters not provided     |

*(Note: `dtype`, `batch_size`, `multiple_of`, `hf_token` were consistent where specified and are omitted for brevity)*

---

### Perplexity Comparison on `tinycompany/ppl` Dataset

| Subset         | Metric | No Global (`Qwentify-No-Global`) | Global + Local (`Qwentify-Global_local`) | ReTok (`Qwentify-ReTok`) |
| :------------- | :----- | :------------------------------- | :--------------------------------------- | :----------------------- |
| **Overall**    | Mean   | 3756.98                          | 2946.80                                  | 4780.19                  |
|                | Median | 6.49                             | 5.98                                     | 6.43                     |
|                | Std    | 7705.66                          | 6007.59                                  | 10041.23                 |
|                | Range  | 1.88 to 29918.16                 | 1.70 to 31436.87                         | 1.88 to 38115.75         |
| **English**    | Mean   | 3.49                             | 3.37                                     | 3.48                     |
|                | Median | 3.23                             | 3.15                                     | 3.18                     |
|                | Std    | 1.16                             | 1.10                                     | 1.16                     |
|                | Range  | 1.88 to 10.22                    | 1.70 to 8.71                             | 1.88 to 10.22            |
| **Code**       | Mean   | 10.14                            | 8.31                                     | 10.19                    |
|                | Median | 6.22                             | 5.74                                     | 6.25                     |
|                | Std    | 9.51                             | 6.35                                     | 9.60                     |
|                | Range  | 2.19 to 60.31                    | 1.95 to 36.57                            | 2.19 to 60.30            |
| **Math**       | Mean   | 8.52                             | 7.12                                     | 8.53                     |
|                | Median | 6.42                             | 5.99                                     | 6.39                     |
|                | Std    | 6.71                             | 4.28                                     | 6.75                     |
|                | Range  | 2.23 to 37.52                    | 2.16 to 23.76                            | 2.23 to 37.69            |
| **Hinglish**   | Mean   | 2385.01                          | 2533.57                                  | 2417.88                  |
|                | Median | 1618.63                          | 1576.19                                  | 1390.85                  |
|                | Std    | 2534.25                          | 2996.30                                  | 2882.50                  |
|                | Range  | 35.32 to 13961.39                | 34.61 to 17177.08                        | 39.67 to 16292.54        |
| **Hindi**      | Mean   | 20170.01                         | 15140.22                                 | 26265.95                 |
|                | Median | 19960.64                         | 15031.15                                 | 26310.19                 |
|                | Std    | 4806.77                          | 4922.56                                  | 6240.46                  |
|                | Range  | 2411.00 to 29918.16              | 962.80 to 31436.87                       | 2924.49 to 38115.75      |