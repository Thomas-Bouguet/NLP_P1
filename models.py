## Libraries

import numpy as np
from numpy.linalg import norm
from gensim.models import word2vec
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score


## Functions

# TODO Try doc2vec
# TODO Try zero shot learning

def cosine_similarity(X,Y):

    if norm(X)*norm(Y) > 0:
        cosine = np.dot(X,Y)/(norm(X)*norm(Y))
    else:
        cosine = 0.
    
    return cosine


# Build word2vec model
def model_word2vec_wordScale(corpus):

    model = word2vec.Word2Vec(corpus, vector_size=100, window=20, min_count=200, workers=4)

    return model

# Vectorize whole text by making mean of vectors of each word.
def model_word2vec_textScale(text,model):

    vec = np.zeros(100)
    i = 0

    for word in text:
        try:
            vec = vec + model.wv[word]
            i = i + 1
        except:
            continue
    
    if i != 0:
        vec = vec / i
    else:
        vec = np.zeros(100)

    return vec

# Return socres of a vectorized request for all vectorized documents
def model_word2vec_scores(vec_req,vec_docs):

    scores = {}

    for k in vec_docs:
        scores[k] = cosine_similarity(vec_req,vec_docs[k])
    
    return scores

# From scores as dic, it returns a list
def model_word2vec_scores_to_list(scores,corpusDicoDocName):

    true_scores = {}

    for req in scores:
        req_true_score = np.zeros(len(corpusDicoDocName))
        for doc in scores[req]:
            posDocId = corpusDicoDocName[doc]
            req_true_score[posDocId] = scores[req][doc]
        true_scores[req] = req_true_score

    return true_scores

# Return results of word2vec model on the whole corpus
def model_word2vec(corpusDocTokenDic,corpusReqTokenDic,corpus):

    model_wordScale = model_word2vec_wordScale(corpus)

    model = {}

    # Vectorize all docs
    docsToVec = {}
    for k in corpusDocTokenDic:
        docsToVec[k] = model_word2vec_textScale(corpusDocTokenDic[k],model_wordScale)

    # Vectorize all requests
    reqsToVec = {}
    for k in corpusReqTokenDic:
        reqsToVec[k] = model_word2vec_textScale(corpusReqTokenDic[k],model_wordScale)

    # Get score of all requests
    for k in reqsToVec:
        model[k] = model_word2vec_scores(reqsToVec[k],docsToVec)

    return model

# Run word2vec on already evaluated results
def model_word2vec_second(results,corpusDicoDocName,corpusDocTokenDic,corpus):

    model_wordScale = model_word2vec_wordScale(corpus)

    model = {}

    # Dictionnary with inverted keys and values of corpusDicoDocName
    doc_from_position = {}
    for k,v in corpusDicoDocName.items():
        doc_from_position[v] = k

    # Get the docs with top results in previous model
    for req in results:
        positions = []
        docs_to_evaluate = []
        for i in results[req]:
            if i > 0:
                positions.append(list(results[req]).index(i))
                docs_to_evaluate.append(doc_from_position[list(results[req]).index(i)])

        # Vectorize each one of those doc
        docs_to_evaluate_vectorized = []
        for doc in docs_to_evaluate:
            doc_content = corpusDocTokenDic[doc]
            docs_to_evaluate_vectorized.append(model_word2vec_textScale(doc_content,model_wordScale))

        req_vectorized = model_word2vec_textScale(req,model_wordScale)

        docs_evaluated = []
        for doc in docs_to_evaluate_vectorized:
            docs_evaluated.append(cosine_similarity(req_vectorized,doc))

        # Re-fill result with 0 for other documents
        res = np.zeros(len(corpusDicoDocName))
        for j in range(len(docs_evaluated)):
            res[positions[j]] = docs_evaluated[j]

        model[req] = res

    return model


# Return results of BM25 model on the whole corpus
def model_bm25(corpusDocTokenList,corpusReqTokenDic):

    bm25 = BM25Okapi(corpusDocTokenList)

    dic_doc_scores = {}
    
    for req in corpusReqTokenDic:
        reqTokenList = corpusReqTokenDic[req]
        doc_scores = bm25.get_scores(reqTokenList)
        dic_doc_scores[req] = doc_scores
    
    return dic_doc_scores


# Return list with only the top_nb number of top results
def top_results(predictions,corpusDicoDocName,top_nb=10):

    results = {}

    for req in predictions:
        list_predictions = list(predictions[req])
        new_d = {}

        for k,v in corpusDicoDocName.items():
            new_d[list_predictions[v]] = k
        
        sorted_list_predictions = list_predictions.copy()
        sorted_list_predictions.sort(reverse=True)

        result = np.zeros(len(predictions[req]))

        i = 0
        for p in sorted_list_predictions:
            result[corpusDicoDocName[new_d[p]]] = p
            i = i + 1
            if i == top_nb:
                break
        
        results[req] = result
    
    return results