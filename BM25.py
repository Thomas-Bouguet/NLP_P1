# Code intéressant pour comprendre BM25, en particulier la structure de code pour l'utiliser.
# Par contre, compliqué pour rien, y a des lignes à enlever, (prendre que l'entrée et la sortie ?)

#!git clone https://github.com/cr-nlp/project1-2023.git

import urllib.request as re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.metrics import ndcg_score
from rank_bm25 import BM25Okapi

dir = "./project1-2023/"
stopword = stopwords.words('english')

def loadNFCorpus():
	filename = dir +"dev.docs"

	# Load data in dev.docs into a dictionnary with id in keys and text in values
	dicDoc={}
	with open(filename) as file:
		lines = file.readlines()
	for line in lines:
		tabLine = line.split('\t')
		#print(tabLine)
		key = tabLine[0]
		value = tabLine[1]
		#print(value)
		dicDoc[key] = value
	filename = dir + "dev.all.queries"
	dicReq={}
	with open(filename) as file:
		lines = file.readlines()
	for line in lines:
		tabLine = line.split('\t')
		key = tabLine[0]
		value = tabLine[1]
		dicReq[key] = value
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

def run_bm25_only(startDoc,endDoc):

	dicDoc, dicReq, dicReqDoc = loadNFCorpus()

	docsToKeep=[]
	reqsToKeep=[]
	dicReqDocToKeep=defaultdict(dict)

	# Gather all documents id, requests id and combinated score in dicReqDocToKeep
	# Also put distinct documents id in docsToKeep and distinct requests id in reqsToKeep
	i=startDoc
	for reqId in dicReqDoc:
		if i > (endDoc - startDoc) :
			break
		for docId in dicReqDoc[reqId]:
			dicReqDocToKeep[reqId][docId] = dicReqDoc[reqId][docId]
			docsToKeep.append(docId)
			i = i + 1
		reqsToKeep.append(reqId)
	docsToKeep = list(set(docsToKeep))


	""" Unused code, may be useful for vocabulary building TODO ?

	# Gather all distinct tokenized words of documents as a vocabulary
	allVocab ={}
	for k in docsToKeep:
		docTokenList = text2TokenList(dicDoc[k])
		#print(docTokenList)
		for word in docTokenList:
			if word not in allVocab:
				allVocab[word] = word
	allVocabListDoc = list(allVocab)
	
	# Gather all distinct tokenized words of requests as a vocabulary
	allVocab ={}
	for k in reqsToKeep:
		docTokenList = text2TokenList(dicReq[k])
		#print(docTokenList)
		for word in docTokenList:
			if word not in allVocab:
				allVocab[word] = word
	allVocabListReq = list(allVocab)

	"""


	corpusDocTokenList = []
	corpusDocName=[]
	corpusDicoDocName={}

	# Store tokzenized documents in a list, names of documents in list and position of the document for the corresponding name in a dictionnary
	i = 0
	for k in docsToKeep:
		docTokenList = text2TokenList(dicDoc[k])
		corpusDocTokenList.append(docTokenList)
		corpusDocName.append(k)
		corpusDicoDocName[k] = i
		i = i + 1

	# Same for requests
	corpusReqName=[]
	corpusReqTokenList = {}
	corpusDicoReqName={}
	i = 0
	for k in reqsToKeep:
		reqTokenList = text2TokenList(dicReq[k])
		corpusReqTokenList[k] = reqTokenList
		corpusReqName.append(k)
		corpusDicoReqName[k] = i
		i = i + 1

	# Apply BM25 to the tokenized documents corpus
	bm25 = BM25Okapi(corpusDocTokenList)

	ndcgBM25Cumul=0
	nbReq=0
	
	# On prend le top 5
	ndcgTop=5

	# For each tokenized request, get the BM25 scores of all documents. Then, get the known true results and calculate the ndcg@5 score.
	for req in corpusReqTokenList:
		reqTokenList = corpusReqTokenList[req]
		doc_scores = bm25.get_scores(reqTokenList)
		trueDocs = np.zeros(len(corpusDocTokenList))

		for docId in corpusDicoDocName:
			if req in dicReqDocToKeep:
				if docId in dicReqDocToKeep[req]:
					posDocId = corpusDicoDocName[docId]
					trueDocs[posDocId] = dicReqDocToKeep[req][docId]
					
		ndcgBM25Cumul = ndcgBM25Cumul + ndcg_score([trueDocs], [doc_scores],k=ndcgTop)
		nbReq = nbReq + 1

	# Calculate the mean ndcg@5 score, print and return it.
	ndcgBM25Cumul = ndcgBM25Cumul / nbReq
	print("ndcg bm25=",ndcgBM25Cumul)
	
	return ndcgBM25Cumul


## To run

#nb_docs = 3192 #all docs
nb_docs = 150 #for tests
run_bm25_only(0,nb_docs)

