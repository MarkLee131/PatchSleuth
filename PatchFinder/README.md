# PatchFinder

As introduced in the paper, PatchFinder involves two phases:

- Initial retrieval of candidate patches: This phase employs a hybrid approach to retrieve candidate patches at the lexical level and the semantic level.
    - Lexical level: We use the TF-IDF algorithm to retrieve candidate patches at the lexical level. You need to run the code within [TF-IDF](./TF-IDF) to retrieve the candidate patches.

    - Semantic level: We implement it at [https://github.com/MarkLee131/CR_score](https://github.com/MarkLee131/CR_score/tree/issta24). You can use the code to retrieve the candidate patches at the semantic level. 

Based on the retrieving results from the two diomenstions, we can fuze the results to get the top-100 candidate patches.

- Re-ranking via fine-tuning: We fine-tune CodeReviewer to re-rank the top-100 candidate patches. You can refer to the code within [Phase-2](./Phase-2) to reuse the code for fine-tuning.

