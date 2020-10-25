'''
https://engineering.talkdesk.com/what-are-sentence-embeddings-and-why-are-they-useful-53ed370b3f35
https://openreview.net/pdf?id=SyK00v5xx
https://nlp.stanford.edu/pubs/glove.pdf
https://nlp.stanford.edu/projects/glove/
https://towardsdatascience.com/paper-summary-evaluation-of-sentence-embeddings-in-downstream-and-linguistic-probing-tasks-5e6a8c63aab1

https://arxiv.org/abs/1908.10084
'''

import math
import sys
import numpy as np
import warnings 
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import RegexpTokenizer
warnings.filterwarnings(action = 'ignore') 


'''
:param path: file with word vectors
:return: dictionary that maps a word to its vector
'''
def preproc(path='word_embeddings//glove.6B//glove.6B.50d.txt'):
    with open(path, 'r', encoding='utf8') as f:
        data = f.readlines()
    return_dict = {}
    for line in data:
        line = line.split()
        word = line[0]
        vec = [float(vi) for vi in line[1:]]
        return_dict[word] = vec
    return return_dict

'''
:param a, b: vectors
:return: float that can be interpreted as the cosine similarity of a and b
'''
def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

'''
:param a, b: vectors
:return: float that can be interpreted as the euclidean distance of a and b
'''
def euclid(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

'''
:param w2v: dict mapping word to vec
:param a: sentence
:return: average of word vectors
'''
def sent_vec(w2v, a):

	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(a.lower())
	vecs = []
	for t in tokens:
		try:
			vecs.append(w2v[t])
		except Exception as e:
			pass
	return [sum(i)/len(i) for i in list(zip(*vecs))]


'''
:param w2v: dict mapping word to vec
:param v: sentence vector
:param a: entity
:param threshold: cossim[euclid] has to be above[below] (cossim >0.4 euclid <6)
:return: bool if best cossim[euclid] of tokens and sentence is above[below] threshold or -1[10]
'''
def cos_sim_sen(w2v, v, a, threshold=0.4): #6

	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(a.lower())
	bsf = -1 #10
	for t in tokens:
		try:
			cossim = cos(w2v[t], v) #euclid(w2v[t], v)
			bsf = cossim if cossim > bsf else bsf #<
		except Exception as e:
			pass
	return bsf > threshold or bsf == -1 #<threshold or bsf==10