
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
| Transtokenizer       | `tinycompany/q2.5-3b-tt-qtk`     | *N/A*  | *N/A*       | *N/A* | Configuration parameters not provided     |


*(Note: `dtype`, `batch_size`, `multiple_of`, `hf_token` were consistent where specified and are omitted for brevity)*

---

### Perplexity Comparison on `tinycompany/ppl` Dataset
| Subset         | Metric | No Global (`Qwentify-No-Global`) | Global + Local (`Qwentify-Global_local`) | ReTok (`Qwentify-ReTok`) | TransTokenizer (`Qwentify-TransTokenizer`) |
| :------------- | :----- | :------------------------------- | :--------------------------------------- | :----------------------- | :----------------------------------------- |
| **Overall**    | Mean   | 3756.98                          | 2946.80                                  | 4780.19                  | 3491.37                                    |
|                | Median | 6.49                             | 5.98                                     | 6.43                     | 335.88                                     |
|                | Std    | 7705.66                          | 6007.59                                  | 10041.23                 | 8127.47                                    |
|                | Range  | 1.88 to 29918.16                 | 1.70 to 31436.87                         | 1.88 to 38115.75         | 20.11 to 68009.48                          |
| **English**    | Mean   | 3.49                             | 3.37                                     | 3.48                     | 250.28                                     |
|                | Median | 3.23                             | 3.15                                     | 3.18                     | 218.41                                     |
|                | Std    | 1.16                             | 1.10                                     | 1.16                     | 162.06                                     |
|                | Range  | 1.88 to 10.22                    | 1.70 to 8.71                             | 1.88 to 10.22            | 48.81 to 1518.07                           |
| **Code**       | Mean   | 10.14                            | 8.31                                     | 10.19                    | 102.50                                     |
|                | Median | 6.22                             | 5.74                                     | 6.25                     | 74.80                                      |
|                | Std    | 9.51                             | 6.35                                     | 9.60                     | 93.13                                      |
|                | Range  | 2.19 to 60.31                    | 1.95 to 36.57                            | 2.19 to 60.30            | 20.11 to 645.39                            |
| **Math**       | Mean   | 8.52                             | 7.12                                     | 8.53                     | 522.06                                     |
|                | Median | 6.42                             | 5.99                                     | 6.39                     | 428.66                                     |
|                | Std    | 6.71                             | 4.28                                     | 6.75                     | 392.45                                     |
|                | Range  | 2.23 to 37.52                    | 2.16 to 23.76                            | 2.23 to 37.69            | 73.91 to 2133.09                           |
| **Hinglish**   | Mean   | 2385.01                          | 2533.57                                  | 2417.88                  | 15700.99                                   |
|                | Median | 1618.63                          | 1576.19                                  | 1390.85                  | 9597.05                                    |
|                | Std    | 2534.25                          | 2996.30                                  | 2882.50                  | 13908.18                                   |
|                | Range  | 35.32 to 13961.39                | 34.61 to 17177.08                        | 39.67 to 16292.54        | 233.87 to 68009.48                         |
| **Hindi**      | Mean   | 20170.01                         | 15140.22                                 | 26265.95                 | 4271.47                                    |
|                | Median | 19960.64                         | 15031.15                                 | 26310.19                 | 3536.75                                    |
|                | Std    | 4806.77                          | 4922.56                                  | 6240.46                  | 3906.28                                    |
|                | Range  | 2411.00 to 29918.16              | 962.80 to 31436.87                       | 2924.49 to 38115.75      | 597.96 to 35859.40                         |
