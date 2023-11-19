## Libraries

import numpy as np
from sklearn.metrics import ndcg_score


## Global variables

# On prend le top 5
ndcgTop=5


## Functions

def ndcgAt5(predictions,dicReqDocToKeep,corpusDicoDocName):

    ndcgBM25Cumul=0
    nbReq=0

    # For each tokenized request, get the BM25 scores of all documents. Then, get the known true results and calculate the ndcg@5 score.
    for req in predictions:
        trueDocs = np.zeros(len(corpusDicoDocName))
        for docId in corpusDicoDocName:
            if req in dicReqDocToKeep:
                if docId in dicReqDocToKeep[req]:
                    posDocId = corpusDicoDocName[docId]
                    trueDocs[posDocId] = dicReqDocToKeep[req][docId]
                    
        ndcgBM25Cumul = ndcgBM25Cumul + ndcg_score([trueDocs],[predictions[req]],k=ndcgTop)
        nbReq = nbReq + 1

    # Calculate the mean ndcg@5 score, print and return it.
    ndcgBM25Cumul = ndcgBM25Cumul / nbReq

    return ndcgBM25Cumul