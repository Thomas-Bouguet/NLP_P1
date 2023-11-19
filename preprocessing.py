## Libraries

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from sklearn.metrics import ndcg_score
from rank_bm25 import BM25Okapi


## Global Variables

dir = "./project1-2023/"
stopword = stopwords.words('english')



# Functions

# TODO Add lemmatizer

def load_data(size=-1):

    # Load data in dev.docs into a dictionnary with id in keys and text in values
    filename = dir +"dev.docs"
    dicDoc={}

    with open(filename) as file:
        lines = file.readlines()

    if size > 0 & size <= len(lines):
        lines = lines[0:size]

    for line in lines:
        tabLine = line.split('\t')
        key = tabLine[0]
        value = tabLine[1]
        dicDoc[key] = value
    
    # Load queries in dev.all.queries into a dictionnary with id of query in key and content in value
    filename = dir + "dev.all.queries"
    dicReq={}

    with open(filename) as file:
        lines = file.readlines()

    if size > 0 & size <= len(lines):
        lines = lines[0:size]
    
    for line in lines:
        tabLine = line.split('\t')
        key = tabLine[0]
        value = tabLine[1]
        dicReq[key] = value
    
    # Load 
    filename = dir + "dev.2-1-0.qrel"
    dicReqDoc=defaultdict(dict)

    with open(filename) as file:
        lines = file.readlines()

    for line in lines:
        tabLine = line.strip().split('\t')

        req = tabLine[0]
        doc = tabLine[2]
        score = int(tabLine[3])
        dicReqDoc[req][doc]=score

    return dicDoc, dicReq, dicReqDoc

def text2TokenList(text):

	# Tokenize and eliminate stopwords of a given text
	word_tokens = word_tokenize(text.lower())
	word_tokens_without_stops = [word for word in word_tokens if word not in stopword and len(word)>2]
	return word_tokens_without_stops

def tokenize_list(givenList,toKeep):

    corpusTokenDic = {}
    corpusTokenList = []
    corpusName=[]
    corpusDicoName={}

    i = 0
    for k in toKeep:
        tokenList = text2TokenList(givenList[k])
        corpusTokenList.append(tokenList)
        corpusTokenDic[k] = tokenList
        corpusName.append(k)
        corpusDicoName[k] = i
        i = i + 1

    return corpusTokenList, corpusTokenDic, corpusName, corpusDicoName


def tokenize(dicDoc, dicReq, dicReqDoc):

    # global docsToKeep,reqsToKeep,dicTrueResults,corpusDocTokenList,corpusDocName,corpusDicoDocName,corpusReqName,corpusReqTokenList,corpusDicoReqName

    docsToKeep = []
    reqsToKeep = []
    dicTrueResults=defaultdict(dict)

    # Gather all documents id, requests id and combinated score in dicTrueResults
    # Also put distinct documents id in docsToKeep and distinct requests id in reqsToKeep
    for reqId in dicReqDoc:
        for docId in dicReqDoc[reqId]:
            dicTrueResults[reqId][docId] = dicReqDoc[reqId][docId]
            docsToKeep.append(docId)
        reqsToKeep.append(reqId)
    docsToKeep = list(set(docsToKeep))

    # Store tokzenized documents in a list, names of documents in list and position of the document for the corresponding name in a dictionnary
    corpusDocTokenList, corpusDocTokenDic, corpusDocName, corpusDicoDocName = tokenize_list(dicDoc,docsToKeep)
    # Same for requests
    corpusReqTokenList, corpusReqTokenDic, corpusReqName, corpusDicoReqName = tokenize_list(dicReq,reqsToKeep)

    # return à mettre dans un dico TODO
    return docsToKeep,reqsToKeep,dicTrueResults,corpusDocTokenList,corpusDocTokenDic,corpusDocName,corpusDicoDocName,corpusReqName,corpusReqTokenList,corpusReqTokenDic,corpusDicoReqName


def build_corpus_dicToList(dic):

    l = []

    for k in dic:
        sentence = dic[k]
        sentence = text2TokenList(sentence)
        l.append(sentence)
    
    return l
    

def build_corpus(dicDoc):
    
    corpus = build_corpus_dicToList(dicDoc)

    return corpus


def get_vocabulary_list(givenList,toKeep):

    allVocab ={}
    for k in toKeep:
        tokenList = text2TokenList(givenList[k])
        for word in tokenList:
            if word not in allVocab:
                allVocab[word] = word
    allVocabList = list(allVocab)

    return allVocabList

def get_vocabulary(dicDoc,dicReq,docToKeep,reqToKeep):

    # Gather all distinct tokenized words of documents as a vocabulary
    allVocabListDoc = get_vocabulary_list(dicDoc,docToKeep)

    # Gather all distinct tokenized words of requests as a vocabulary
    allVocabListReq = get_vocabulary_list(dicReq,reqToKeep)

    return allVocabListReq, allVocabListDoc