# https://pypi.org/project/stanford-openie

from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from openie import StanfordOpenIE
from pprint import pprint
import requests
from bs4 import BeautifulSoup as soup
from googlesearch import search
import time
import networkx as nx
import matplotlib.pyplot as plt
import warnings 
from tqdm import tqdm
warnings.filterwarnings(action = 'ignore') 

'''
:param graph: nx graph to enhance with new knowledge
:param triples: statements to enhynce graoh with
:param relation: label of edges connecting the graoh with new nodes
:return: new nodes added to the graph
'''
def add_triples_to_graph(graph, triples, relation='=openie='):

	'''
	:param entity: node from old graph
	:param subject: node from new graph
	:return: bool if nodes share one token
	'''
	def match(entity='', subject=''):
		
		tokenizer = RegexpTokenizer(r'\w+') 
		lemmatizer = WordNetLemmatizer()
		entity = [lemmatizer.lemmatize(e) for e in tokenizer.tokenize(remove_stopwords(entity).lower()) if all([not e.isnumeric(), not e[0].isnumeric(), len(e)>2])]
		subject = [lemmatizer.lemmatize(e) for e in tokenizer.tokenize(remove_stopwords(subject).lower()) if all([not e.isnumeric(), not e[0].isnumeric(), len(e)>2])]

		return not all([e not in subject for e in entity])

	# connect new subjects and objects to graph
	for n,d in tqdm(list(graph.nodes.data())):
		for s,p,o in triples:
			if match(d['name'], s):
				graph.add_edge(n, s, key=relation)
			if match(d['name'], o):
				graph.add_edge(o, n, key=relation)

	# enhance graph with new edges
	new_nodes = []
	for s,p,o in triples:
		graph.add_node(s, name=s)
		graph.add_node(o, name=o)
		graph.add_edge(s, o, key=p)
		new_nodes.append(s)
		new_nodes.append(o)

	return new_nodes

'''
:param corpus: string to annotate openie on
:return: list of statements (s,p,o) to build the graph on
'''
def get_triples(corpus=''):
	try:
		with StanfordOpenIE() as client:
			return list(dict.fromkeys([(t['subject'], t['relation'], t['object']) for t in client.annotate(corpus)]))
	except:
		return []

'''
:param urls: urls to tae the texts from
:param max_chars: annotation limit of openie
:param min_chars: no need to reach upper limit so lower limit is enough
:return: corpus to annotate openie on
'''
def get_corpus(urls=[], max_chars=99999, min_chars=70000):
	docs = []
	for u in tqdm(urls):
		response = requests.get(u)
		if response:
			raw = response.text
			html = soup(raw, 'html.parser')
			docs.extend([t for t in html.stripped_strings if len(t.split())>2])
		else:
			print('URL Error:', u)

	docs = sorted(docs, key=lambda doc: len(doc), reverse=True)

	corpus = ''
	for doc in docs:
		if len(corpus) + len(doc) <= max_chars:
			corpus = '. '.join([corpus, doc])
		if len(corpus) > min_chars:
			break
	return corpus

'''
:param query: string to search on google
:param max_docs: number of docs to consider
:param max_triples: number of statements to consider
:return: statments (s,p,o) to build the graoh on
'''
def google_openie(query='', max_docs=3, max_triples=600):
	hrefs = search(query=query, tld='com', lang='en', num=max_docs, start=0, stop=max_docs, pause=2)
	corpus = get_corpus(hrefs)
	triples = get_triples(corpus)
	# todo rank
	return triples[:max_triples]

#______________________________________________________________________________


# def temp_add_triples_to_graph(graph, triples, relation='=openie='):

# 	def match(entity, subject):
# 		entity = entity.split()
# 		entity = [s.rstrip().rstrip('(').rstrip(')') for s in entity]
# 		entity = [s for s in entity if len(s)>2]
# 		subject = subject.split()
# 		subject = [s.rstrip().rstrip('(').rstrip(')') for s in subject]
# 		subject = [s for s in subject if len(s)>2]

# 		for e in entity:
# 			for s in subject:
# 				if e.lower() in s.lower() or s.lower() in e.lower():
# 					return True
# 		return False

# 	for n,d in list(graph.nodes.data()):
# 		for s,p,o in triples:
# 			if match(n, s):
# 				graph.add_edge(n, s, label=relation)
# 			if match(n, o):
# 				graph.add_edge(o, n, label=relation)

# 	trips = []

# 	for s,p,o in triples:
# 		graph.add_node(s, name=s)
# 		graph.add_node(o, name=o)
# 		graph.add_edge(s, o, key=p)

# 		trips.append(s)
# 		trips.append(o)

# 	return trips


# if __name__ == '__main__':
# 	print('===== OpenIE on GOOGLE =====')
# 	starttime = time.time()

# 	trips = google_openie(query=some_topic, max_docs=3, max_triples=800)
# 	print('QUERY:', some_topic)
# 	print('TRIPLES:', len(trips))
# 	# for t in triples:
# 	# 	print(' => '.join(t.values()))


# 	G = nx.DiGraph()
# 	for s,p,o in trips:
# 		G.add_node(s)
# 		G.add_node(o)
# 		G.add_edge(s,o, label=p)

# 	combinations = []
# 	for s in some_sen.split():
# 		for ss in some_sen.split():
# 			s = s.lower()
# 			ss = ss.lower()
# 			if s != ss and s not in stopwords.words('english') and ss not in stopwords.words('english') and (ss,s) not in combinations  and (s,ss) not in combinations and len(s) > 2 and len(ss) > 2:
# 				combinations.append((s, ss))
# 	startnodes = [sn for sn,_ in combinations]
# 	startnodes = list(dict.fromkeys(startnodes))
# 	print(startnodes)

# 	for sn in startnodes:
# 		G.add_node(sn)

# 	temp_add_triples_to_graph(G, trips)
# 	print('graph nodes:', len(G.nodes))
# 	print('graph edges:', len(G.edges))


# 	paths = []
# 	for start, end in combinations:
# 		try:
# 			path = nx.shortest_path(G, start, end)
# 			paths.append(path)
# 		except:
# 			pass

# 	for p in paths:
# 		print(' => '.join(p))
# 	print('found paths:', len(paths))


	print('RUNTIME:', time.time()-starttime, 'sec')

	# pos = nx.spring_layout(G)
	# nx.draw_networkx_nodes(G, pos, node_color='k', node_size=1)
	# nx.draw_networkx_edges(G, pos)
	# nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels={(n1, n2): l['label'] for n1, n2, l in list(G.edges.data())})
	# nx.draw_networkx_labels(G, pos, font_size=6)
	# plt.axis('off')
	# plt.show()



