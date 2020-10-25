import numpy as np 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import json
from pprint import pprint
from gensim.parsing.preprocessing import STOPWORDS
from matplotlib.ticker import MaxNLocator
from entity_properties.wikiapi import get_text
import timeit

STOPWORDS = list(STOPWORDS)
STOPWORDS.extend('add pp new ed isbn year time'.split())

with open('entity_properties/property_blacklist.txt') as f:
	prop_blacklist = f.readlines()
	prop_blacklist = [p.rstrip() for p in prop_blacklist]

with open('entity_properties/property_frequencies.json') as f:
	data = json.load(f)	

'''
TFIDF = tf(t,d)*log(N/(df+1)) => http://www.tfidf.com/
tf(t,d) = count t in d / number of words in d
df(t) = occ of t in docs

1. build a matrix with tfidf of every word-property pair
2. add up the row for each word
3. rank the words by their cumulated scores

:param word: list fo strings to rank by their tfidf referring the property descriptions
:return: list of strings
'''
def word_rankings(words=[]):

	ranking = dict.fromkeys(words)
	num_docs = len(data.items()) #N
	num_docs_with_word = dict.fromkeys(words, 0) #occ of t in docs
	num_words_in_doc = {} #number of words in d
	num_word_in_doc = {} #count t in d
	for k, v in data.items(): #iterate property descriptions
		doc_words = v['aliases'].split()
		doc_words.extend(v['description'].split())
		doc_words.extend(v['label'].split())
		num_words_in_doc[k] = len(doc_words)
		num_word_in_doc[k] = dict.fromkeys(words)
		for word in words:
			count = len([w for w in doc_words if ' '+word in ' '+w])
			num_word_in_doc[k][word] = count
			if count > 0:
				num_docs_with_word[word] += 1

	for w in words: #calculate scores and add them up for every word
		tfidf_list = [(num_word_in_doc[k][w]/num_words_in_doc[k]) * np.log(num_docs/(num_docs_with_word[w]+1)) for k, v in data.items()]
		ranking[w] = sum(tfidf_list)

	return_list = [(k, v) for k, v in ranking.items()]
	return_list = sorted(return_list, key=lambda i: -i[1]) #rank them by cumulated scores
	
	return return_list

'''
1. preprocess documents
2. build dictionary and corpus
3. train lda model
4. get top (cnt many) topics

:param docs: list of articles building the corpus to train lda on
:param cnt_topics: number of topics lda should consider
:return: list of topics represented as a lsit of words (spec. a word,importance-tuple) each 
'''
def load_top_topics(docs=[], cnt_topics=5):

	docs = [remove_stopwords(doc) for doc in docs] #remove stopwords
	tokenizer = RegexpTokenizer(r'\w+') #=> https://www.kite.com/python/docs/nltk.RegexpTokenizer
	for i in range(len(docs)):
		docs[i] = docs[i].lower() #lower strings
		docs[i] = tokenizer.tokenize(docs[i]) #split strings into tokens
	docs = [[token for token in doc if not token.isnumeric() and not token[0].isnumeric()] for doc in docs] #exclude numbers
	docs = [[token for token in doc if len(token) > 1] for doc in docs] #exclude too short tokens
	lemmatizer = WordNetLemmatizer() #=> https://www.nltk.org/_modules/nltk/stem/wordnet.html
	docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs] #group similar words

	dictionary = Dictionary(docs) #create dictionary
	corpus = [dictionary.doc2bow(doc) for doc in docs] #create corpus

	model = LdaModel( #=> https://radimrehurek.com/gensim/models/ldamodel.html
		corpus=corpus,
		id2word=dictionary,
		# chunksize=2000,
		# alpha='auto',
		# eta='auto',
		iterations=200,
		# passes=20,
		# eval_every=None,
		num_topics=cnt_topics
	)
		
	top_topics = model.top_topics(corpus) #[([(a, x), ..., (a, x)], a), ...]

	return top_topics

'''
1. iterate property descriptions and consider every property where w appears
2. check conditions
	- above count_threshold
	- not in blacklist,
	- correct property-type (not a string, quantitiy, ...)
	- not an ID
	- first letter no capital

:param count_threshold: how often property should appear on wikidata
:param exclude_capitalletter: most withfirst latter being capital are useless
:param exclude_blacklist: 
:return: properties list of tuples where each tuple is of type ('P26', 'spouse', 97474)
'''
def get_properties(w='', count_threshold=1000, exclude_capitalLetter=True, exclude_blacklist=True):

	w_space = ' ' + w
	prop_ids = []
	for k, v in data.items(): 

		if ( w_space in v['aliases'] or w_space in v['description'] or w_space in v['label'] ) and v['count'] > count_threshold:
			if (exclude_blacklist) and (k not in prop_blacklist) and (v['data-type'] == 'wikibase-item'):
				if ' ID' not in v['label']:
					if (exclude_capitalLetter) and (v['label'][0].islower()):
						prop_ids.append((k, v['label'], v['count']))

	return prop_ids

'''
1. load wikipedia articles as corpus
2. train lda to extract top topics as list of words
3. rank word via tfidf referring properties of wikidata
4. find related properties for most important words
5. take best properties regarding number of appearances on wikidata

:param entities: list of strings to find best properties for
:param tfidf_threshold: tfidf score property has in frequencies-file
:param count_threshold: how often property should appear on wikidata
:param cnt_topics: number of topics lda should consider
:param exclude_capitalletter: most withfirst latter being capital are useless
:param exclude_blacklist: 
:param max_props: max number of properties to return
:return: properties list of tuples where each tuple is of tyoe ('P26', 'spouse')
'''
def props_per_entities(entities=[], tfidf_threshold=2.5, count_threshold=1000, cnt_topics=5, 
	exclude_capitalLetter=True, exclude_blacklist=True, max_props=50):

	docs = [get_text(e) for e in entities] #get wikipedia articles
	
	top_topics = load_top_topics(docs=docs, cnt_topics=cnt_topics) #get topics (cnt many) referring docs via lda

	top_words = [] #each topic is represented by words so collect them and exclude stopwords
	for (topic, _) in top_topics:
		for (p, w) in topic:
			if w not in STOPWORDS:
				top_words.append(w)
	top_words = list(dict.fromkeys(top_words)) #exclude dublicates
	word_rank = word_rankings(top_words) #rank words by their tfidf
	top_words = [w for w,v in word_rank if v > tfidf_threshold] #cut at threshold

	properties = [] #each word relates t oseveral properties so collect them and exclude bad ones
	for w in top_words:
		props = get_properties(w=w, count_threshold=count_threshold, exclude_capitalLetter=exclude_capitalLetter, exclude_blacklist=exclude_blacklist)
		if len(props) > 0:
			properties.extend(props)
	properties = list(dict.fromkeys(properties)) #exclude dublicates

	properties = sorted(properties, key=lambda p: -p[2]) #rank properties by their number of appearances on wikidata
	properties = [(i, l) for (i, l, c) in properties] #exclude unnecessary information
	properties = properties[:max_props] #cut at threshold

	return properties