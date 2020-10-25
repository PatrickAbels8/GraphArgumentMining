from nltk.tokenize import word_tokenize
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import spacy
import time

def retrieve_wordnet_candidates(mention, limit, pure_tokens, sentence_tokens, sentence, mentionContextSize):
    sparql = SPARQLWrapper("http://knowledge-graph:8890/sparql")
    candidates = []
    candidate_ids = []
    delete_candidates = []
    # First step: Do full text search
    sparql.setQuery("""
        SELECT DISTINCT ?s ?label_1 ?label_2
        FROM <http://wordnet-rdf.princeton.edu>
        WHERE {
        ?s ?label_1 ?label_2.
        ?label_2 bif:contains " """ + mention + ' ". '
        + """
        } LIMIT """ + limit)

    sparql.setReturnFormat(JSON)
    results = query_virtuoso(sparql)
        
    mention = mention.strip("'")

    # Second steps: Get node IDs:
    mention_context, orig_mention = get_context(mention, pure_tokens, sentence_tokens, sentence,
                                                mentionContextSize)
    for result in results["results"]["bindings"]:
        node = result["s"]["value"]
        label = result["label_2"]["value"]

        # compute edit distance and keep only top 20 candidates!
        candidate = WordnetCandidateEntity(node, label)
        candidates.append(candidate)

        candidate.levMatchLabel = levenshtein(candidate.label.lower(), orig_mention.lower())
        candidate.levMatchContext = levenshtein(candidate.label.lower(), mention_context.lower())

    cdx = rank_importance(candidates)
    candidates = cdx[:15]

    # Third step: Get Lemmas from NodeIDs
    for candidate in candidates:
        node = "<" + str(candidate.nodeID) + ">"
        sparql.setQuery("""
        SELECT DISTINCT ?s ?label_1
        FROM <http://wordnet-rdf.princeton.edu>
        WHERE {
        ?s ?label_1 """ + node + """.
        }
        """)
        sparql.setReturnFormat(JSON)
        node_results = query_virtuoso(sparql)

        lemmas = []
        for result in node_results["results"]["bindings"]:
            # If result already is an ID:
            if "/id/" in result["s"]["value"]:
                candidateID = "<" + result["s"]["value"] + ">"     
                if candidateID not in candidate_ids:   
                    candidate.candidateID = candidateID
                    candidate_ids.append(candidate.candidateID)
                else:
                    candidate.candidateID = "none"
            else:
                lemma = result["s"]["value"]
                lemmas.append(lemma)
                candidate.lemma = lemma

        # if the candidate has no ID yet, find the ID!
        if candidate.candidateID == None:
            lemma = "<" + candidate.lemma + ">"
            sparql.setQuery("""
                    SELECT DISTINCT ?label_1 ?label_2
                    FROM <http://wordnet-rdf.princeton.edu>
                    WHERE {
                     """ + lemma + """ ?label_1 ?label_2.
                    } """)
            sparql.setReturnFormat(JSON)
            lemma_results = query_virtuoso(sparql)

            mention = mention.replace(" ", "_")
            for result in lemma_results["results"]["bindings"]:
                if mention in result["label_2"]["value"].lower():
                    try:
                        splitted = result["label_2"]["value"].split("-")
                        candidateID = splitted[-2] + "-" + splitted[-1]
                        url = "<http://wordnet-rdf.princeton.edu/id/" + candidateID + ">"
                        if url not in candidate_ids:
                            candidate_ids.append(url)
                            candidate.candidateID = url
                            break
                    except:
                        candidate.candidateID = "none"

        if candidate.candidateID == None or candidate.candidateID == "none":
            delete_candidates.append(candidate)

    for del_can in delete_candidates:
        can_index = candidates.index(del_can)
        del candidates[can_index]
        
    return candidates


# retrieves a list of candidates from a given knowledge base
def retrieve_candidates(mention, knowledge_base, language, limit, pure_tokens, sentence_tokens, sentence, sentenceContextSize):
    if knowledge_base == "wordnet":
        candidates = retrieve_wordnet_candidates(mention, limit, pure_tokens, sentence_tokens, sentence, sentenceContextSize)
        return candidates

    sparql = SPARQLWrapper("http://knowledge-graph:8890/sparql")
    candidates = []
    
    # set the language filter
    if language == "de":
        language_filter = '?concept rdfs:label ?label. FILTER ( lang(?label) = "de" || lang(?label) = "de-ch" || lang(?label) = "de-at")'
    if language == "en":
        language_filter = '?concept rdfs:label ?label. FILTER ( lang(?label) = "en-gb" || lang(?label) = "en-ca" || lang(?label) = "en")'
    else:
        language_filter = ""

    start = time.time()

    graph = ""
    # set the graph url
    if knowledge_base == "wikidata":
        graph = "<http://www.wikidata.org>"

    if knowledge_base == "wikidata_truthy":
        graph = "<https://wikidata.org>"

    sparql.setQuery("""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT DISTINCT ?concept ?altLabel ?label ?description
    FROM """ + graph + """
    WHERE {
    VALUES ?labelpredicate {rdfs:label skos:altLabel}
    {
    ?concept ?labelpredicate ?altLabel.
    ?altLabel bif:contains " """ + mention + ' ". '
    + language_filter + """
    }
    } LIMIT """ + limit )

    if knowledge_base == "nell":
        graph = "<http://rtw.ml.cmu.edu/rtw/kbbrowser>"

        sparql.setQuery("""
                SELECT DISTINCT ?concept ?altLabel
                FROM """ + graph + """
                WHERE {
                ?concept ?predicate ?altLabel.
                ?altLabel bif:contains " """ + mention + ' ". ' + """
                }
                LIMIT """ + limit)


    sparql.setReturnFormat(JSON)
    results = query_virtuoso(sparql)
    if(knowledge_base == "wikidata" or knowledge_base == "wikidata_truthy"):
        for result in results["results"]["bindings"]:
            #print(result)
            try:
                label = result['label']
            except KeyError:
                label = result["altLabel"]
            try:
                label_2 = result["altLabel"]
            except KeyError:
                label_2 = result["label"]
            candidate = CandidateEntity(result["concept"]["value"].split("/")[-1], label['value'], label_2["value"], label_2["xml:lang"] )
            if language in label_2["xml:lang"]:
                candidates.append(candidate)
    elif(knowledge_base == "nell"):
        for result in results["results"]["bindings"]:
            # filter out relation tokens:
            if("Token_" in result["concept"]["value"]):
                continue
            # filter out too long labels:
            if (len(result["altLabel"]["value"]) > 200):
                continue
            if("FROM:" in result["altLabel"]["value"]):
                continue
            if("Execution_MBL" in result["concept"]["value"]):
                continue
            if("Pattern_" in result["concept"]["value"]):
                continue
            identifier = result["concept"]["value"].split("/")[-1]
            category = identifier.split("_")[0]
            entity = identifier.replace(category + "_", "")
            iri = "concept:" + category + ":" + entity
            candidate = CandidateEntity(iri, result["altLabel"]["value"], "-" ,"en")
            candidates.append(candidate)
    time_taken = time.time() - start

    #print("Retrieve candidates took: ", time_taken)

    return candidates

def query_virtuoso(sparql):
    query_counter = 0
    max_retries = 20
    delay = 10
    while query_counter < max_retries:
        try:
            results = sparql.query().convert()
            query_counter = max_retries
        except Exception as he:
            query_counter += 1
            print("Error is: ", he)
            print("HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)
    return results

class WordnetCandidateEntity:
    def __init__(self, aNodeID, aLabel):
        self.label = aLabel
        self.nodeID = aNodeID

    candidateID = None
    lemma = None
    levMatchLabel = 0  # edit distance between mention and candidate entity label
    levContext = 0  # edit distance between mention + context and candidate entity label
    signatureOverlap = set() # set of directly related entities as IRI Strings
    numRelatedRelations = 0 # number of distinct relations to other entities
    signatureOverlapScore = 0 # number of related entities whose entity label occurs in <i>content tokens</i> <i>Content tokens</i> consist of tokens in mention sentence annotated as nouns, verbs or adjectives
    idRank = 0 # logarithm of the wikidata ID - based on the assumption that lower IDs are more important


class CandidateEntity:
    def __init__(self, anIRI, aLabel, anAlternativeLabel, aLanguage):
        self.iri = anIRI # the IRI String of this entity
        self.label = aLabel # the main label of this entity
        self.alternativeLabel = anAlternativeLabel #An alternative label (alias) of this entity
        self.language = aLanguage # language of this candidate entry
        self.frequency = 0 # in-link count of wikipedia article of IRI
    
    levMatchLabel = 0 # edit distance between mention and candidate entity label
    levContext = 0 # edit distance between mention + context and candidate entity label
    signatureOverlap = set() # set of directly related entities as IRI Strings
    numRelatedRelations = 0 # number of distinct relations to other entities
    signatureOverlapScore = 0 # number of related entities whose entity label occurs in <i>content tokens</i> <i>Content tokens</i> consist of tokens in mention sentence annotated as nouns, verbs or adjectives
    idRank = 0 # logarithm of the wikidata ID - based on the assumption that lower IDs are more important  
    relatedEntities = set()
    relatedRelations = set()

# sentence: String, the sentence to be lemmatized
# lemmatizing: Boolean, do lemmatizing true/false
def lemmatize_sentence(sentence, lemmatizing):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    lemmatized_tokens = []
    doc = nlp(sentence)
    sentence_tokens = []
    sentence_lemmatized = ""
    for token in doc:
        sentence_tokens.append(token.text)
        if lemmatizing:
            lemmatized_tokens.append(token.lemma_)
            sentence_lemmatized = sentence_lemmatized + " " + str(token.lemma_)
    if not lemmatizing:
        sentence_lemmatized = sentence
        lemmatized_tokens = sentence_tokens
    return sentence_tokens, sentence_lemmatized, lemmatized_tokens

def get_context(mention, pure_tokens, sentence_tokens, sentence, mentionContextSize):
    mention_length = len(mention.split(" "))
    mention_pure = ''.join(ch for ch in mention if ch.isalnum())
    if mention_length == 1:
        try:
            index = pure_tokens.index(mention_pure.lower())
            orig_mention = sentence_tokens[index]
        except:
            try:
                orig_mention = sentence_tokens[pure_tokens.index(mention_pure.title())]
            except:
                try:
                    for i,t in enumerate(pure_tokens):
                        if mention_pure in t:
                            #print("mention", mention, "found in: ", t)
                            orig_mention = sentence_tokens[i]
                except Exception as e:
                    print(repr(e))
                    print("Failing to find: ", repr(mention), " in", pure_tokens)
                    print("Sentence was: ", sentence)
    else:
        #print("mention is: ", mention)
        orig_mention = ""
        mention_splitted = mention.split(" ")
        for i in range(mention_length):
            mention_pure = "".join(ch for ch in mention_splitted[i] if ch.isalnum())
            try:
                orig_mention += sentence_tokens[pure_tokens.index(mention_pure.lower())] + " "
            except:
                try:
                    orig_mention += sentence_tokens[pure_tokens.index(mention_pure.title())] + " "
                except:
                    print("mention_splitted_i is: ", mention_splitted[i])
                    #print(pure_tokens)
                    #print(sentence_tokens)
                    for j,t in enumerate(pure_tokens):
                        if mention_splitted[i] in t:
                            #print("mention",mention,"found in:", t)
                            orig_mention += sentence_tokens[j]
                        elif mention.title() in t:
                            orig_mention += sentence_tokens[j]
        if orig_mention == "":
            try:
                orig_mention = sentence_tokens[pure_tokens.index(mention.title().replace(" ",""))]
            except Exception as e:
                 print("Exception here:")
                 print(repr(e))
                 print("orig mention is empty, mention was", mention)
                 print(sentence_tokens)
                 print(pure_tokens)
# get the context:
    #print("orig_mention is: ", orig_mention, "mention is: ", mention)
    #print(mention)    
    #print(sentence_tokens)
    split_context = re.split('(' + re.escape(orig_mention) + ')', sentence)
    left_context = " ".join(word_tokenize(split_context[0])[-mentionContextSize:])
    try:
        right_context = " ".join(word_tokenize(split_context[2])[:min(len(split_context[2]), mentionContextSize)])
    except:
        right_context = ""
    mention_context = left_context + " " + orig_mention + " " + right_context

    return mention_context, orig_mention

# Signature Overlap Score (desc)
# Sum of both Edit Distances (asc)
# Number of distinct relations (desc)
# ID Rank (asc)
def rank_importance(candidates):
    cdx=sorted(candidates,key=lambda x: x.idRank)
    cdx=sorted(cdx, key=lambda x: -x.numRelatedRelations)
    cdx=sorted(cdx, key=lambda x: x.levMatchContext)
    cdx=sorted(cdx, key=lambda x: x.levMatchLabel)
    cdx=sorted(cdx, key=lambda x: -x.signatureOverlapScore)
    return cdx

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# It is the number of related entities whose concept label occurs in the tokens
# of the sentence in which the mention occurs and is not a stopword and does not occur in the mention.
def computeSignatureOverlapScore(can, relatedEntities, sentence_tokens, mention, stopWords):
    punct = set([",", ";", ".", "!", "?", "`", "'", '"', ":" , "-", "_", "+", "#", "*"])
    #start = time.time()
    score = 0
    for entity in relatedEntities:
        entity_tokens = word_tokenize(entity)
        for et in entity_tokens:
            if (et not in stopWords) and (et not in punct) and (et not in mention):
                if et in sentence_tokens:
                    score += 1
                    continue
    #elapsed_time = (time.time() - start)
    #print("computeSignatureOverlapScore:", elapsed_time)
    return score

def get_semantic_signature(identifier, language):
    sparql = SPARQLWrapper("http://knowledge-graph:8890/sparql")
    language = '"' + language + '"'
    identifier = "<" + identifier + ">"
    sparql.setQuery("""
    PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?label ?p
    FROM <http://www.wikidata.org>
    WHERE
    {
    { ?e1 ?rd ?m . ?m ?p """ + identifier + """ . }
    UNION
    { """ + identifier + """ ?p ?m . ?m ?rr ?e1 . }
    ?e1 rdfs:label ?label. FILTER ( lang(?label) =  """ + language + """ )
    } 
    LIMIT 50
    """)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    #print("Found " + str(len(results['results']['bindings'])) + " Semantic entities/relations.")
    
    entities = set()
    relations = set()
    
    for x in sorted(results['results']['bindings'], key=lambda x: x["label"]["value"]):
        entities.add(x['label']['value'])
        relations.add(x['p']['value'])

    return entities, relations
