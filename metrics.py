## Libraries

import numpy as np
from sklearn.metrics import ndcg_score


## Global variables

# On prend le top 5
ndcgTop=5


## Functions

def ndcgAt5(predictions,trueResults,corpusDicoDocName):

    # ndcgBM25Cumul=0
    # nbReq=0

    # # For each tokenized request, get the BM25 scores of all documents. Then, get the known true results and calculate the ndcg@5 score.
    # for req in predictions:
    #     trueDocs = np.zeros(len(corpusDicoDocName))
    #     for docId in corpusDicoDocName:
    #         if req in trueResults:
    #             if docId in trueResults[req]:
    #                 posDocId = corpusDicoDocName[docId]
    #                 trueDocs[posDocId] = trueResults[req][docId]
                    
    #     ndcgBM25Cumul = ndcgBM25Cumul + ndcg_score([trueDocs],[predictions[req]],k=ndcgTop)
    #     nbReq = nbReq + 1

    # # Calculate the mean ndcg@5 score and return it.
    # ndcgBM25Cumul = ndcgBM25Cumul / nbReq
    
    trueDocs = np.zeros(len(corpusDicoDocName))
    tot = 0
    count = 0

    for r in predictions:
        for d in trueResults[r]:
            p = corpusDicoDocName[d]
            trueDocs[p] = trueResults[r][d]

        tot = tot + ndcg_score([trueDocs],[predictions[r]],k=5)
        count = count + 1

    tot = tot / count

    return tot
    # return ndcgBM25Cumul