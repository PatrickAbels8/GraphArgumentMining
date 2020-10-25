from datetime import datetime
import time
import numpy as np
import spotlight
import urllib.parse, urllib.request
import urllib
from helper import handle_urllib_request
import string
from nltk.corpus import stopwords
import entity_linking_helper

def wikifier(text, userkey, lang="en", threshold=0.8, counter=0, topic=False, printouts=True, limit=5, apply_page_rank_sq_threshold=True, **kwargs): #http://wikifier.org/info.html, more: https://en.wikipedia.org/wiki/Entity_linking
    # Prepare the URL.
    wikifier_query = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", userkey),
        ("pageRankSqThreshold", "%g" % threshold),
        ("applyPageRankSqThreshold", "true" if apply_page_rank_sq_threshold == True else False), # do not apply the threshold to prune the list => done below
        ("nTopDfValuesToIgnore", "200"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"),
        ("includeCosines", "false"),  # cos(sent, wiki_id_doc)
        ("maxMentionEntropy", "-1"),
        ("maxTargetsPerMention", "20"),
        ("support", "true")
        ])
    url = "http://www.wikifier.org/annotate-article"

    # Call the Wikifier and read the response.
    response = handle_urllib_request(wikifier_query, url)
    # print(response)

    # Output the annotations.
    results = []
    for i, anno in enumerate(response['annotations']):
        try:
            support_word = text[anno['support'][0]['chFrom']:anno['support'][0]['chTo']+1]
            only_punctuation_as_support = True if sum([0 if char in string.punctuation+str('’') else 1 for char in support_word ]) == 0 else False

            # if support word has only punctuation, drop it. If it is a stopword, drop it, too
            if only_punctuation_as_support == False and support_word.lower() not in stopwords.words('english'):
                #stop_words.ENGLISH_STOP_WORDS => sklearn
                results.append((anno['wikiDataItemId'],
                                      anno["title"],
                                      str(counter+i),
                                      support_word,
                                      anno['pageRank']))
        except KeyError as e:
            print(e)
            pass

    results = sorted(results, key=lambda tup: tup[4], reverse=True)[:limit]


    if printouts == True:
        if topic == False:
            print("\n========== Wikifier annotations found ==========")
        else:
            print("\n========== Wikifier annotations found for topic ==========")

        for id, title, _, support, _ in results:
            print("%s => %s (%s)" % (support, title, id))


    return results, counter+len(response['annotations'])

def dbpedia_spotlight(text, limit=5, lang="en", counter=0, knowledge="wikidata", **kwargs):
    sentence = text
    print(datetime.now())
    signature = False
    lemmatizing = True
    knowledge_base = knowledge
    candidates_per_sentence = limit
    mentionContextSize = 3
    candidateQueryLimit = 300

    sentence_tokens, sentence_lemmatized, lemmatized_tokens = entity_linking_helper.lemmatize_sentence(sentence, lemmatizing)
    pure_tokens = []
    # print("lemmatized_tokens are:", lemmatized_tokens)
    for token in lemmatized_tokens:
        pure_tokens.append(''.join(ch for ch in token if ch.isalnum()))
    # print("Pure tokens are: ", pure_tokens)
    # annotate the sentence with DBPedia Spotlight
    try:
        query_counter = 0
        max_retries = 20
        delay = 10
        while query_counter < max_retries:
            try:
                spotlight_mentions = spotlight.annotate('http://knowledge-graph:2222/rest/annotate',
                                                        sentence_lemmatized)
                query_counter = max_retries
            except Exception as he:
                if "No Resources found" in he:
                    print("No resources found in: ", sentence)
                    return [], counter

                query_counter += 1
                print(
                    "HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
                if query_counter == max_retries:
                    exit(1)
                else:
                    time.sleep(delay)

        spotted_mentions = [mention['surfaceForm'] for mention in spotlight_mentions]
        # TODO: works only for english now!
        spotted_mentions = [mention for mention in spotted_mentions if
                            mention.lower() not in stopwords.words('english') and mention[0].isalnum()]
        # no duplicates of mentions
        spotted_mentions = set(spotted_mentions)
    # print(spotted_mentions)
    except Exception as e:
        print(repr(e))
        print("Could not find any mentions for the sentence: ", sentence)
        return [], counter

    results_all = []  # List of the top results of every mention
    # use the dbpedia spotlight annotations for dbpedia
    if knowledge_base == "dbpedia":
        # use the spotlight_mentions directly, skip rest
        uris = set()
        for result in spotlight_mentions:
            mention = result['surfaceForm']
            annotation = result['URI']
            score = result['similarityScore']
            if annotation not in uris:
                # TODO: set label correctly!
                results_all.append(
                    {"mention": mention, "mention_annotation": [{"label": "?", "id": annotation, "score": score}]})
                uris.add(annotation)
    else:
        for mention in spotted_mentions:
            mention_result = {}

            # print(mention)
            # print(sentence_tokens)
            # print(lemmatized_tokens)
            mention_context, orig_mention = entity_linking_helper.get_context(mention, pure_tokens, sentence_tokens, sentence,
                                                                  mentionContextSize)
            mention_result["mention"] = orig_mention
            mention_query = mention.replace("'", ".")
            mention_query = "'" + mention_query.lower() + "'"

            candidates = entity_linking_helper.retrieve_candidates(mention_query, knowledge_base, lang, str(candidateQueryLimit),
                                                       pure_tokens, sentence_tokens, sentence, mentionContextSize)
            # compute Features, the wordnet features are already commputed
            if knowledge_base != "wordnet":
                for c in candidates:
                    c.levMatchLabel = entity_linking_helper.levenshtein(c.label.lower(), orig_mention.lower())
                    c.levMatchContext = entity_linking_helper.levenshtein(c.label.lower(), mention_context.lower())
                    try:
                        qid = c.iri.split("Q")[1]
                        c.idRank = np.log(int(qid))
                    except:
                        # TODO: what to do with properties found as candidates??
                        pass
            cdx = entity_linking_helper.rank_importance(candidates)
            if knowledge_base == "wordnet":
                mention_result["mention_annotation"] = [{"label": s.label, "id": s.candidateID} for s in
                                                        cdx[:candidates_per_sentence]]
            else:
                mention_result["mention_annotation"] = [{"label": s.label, "id": s.iri} for s in
                                                        cdx[:candidates_per_sentence]]
            top_candidates = cdx[:10]

            if signature and knowledge_base == "wikidata":
                for c in top_candidates:
                    entities, relations = entity_linking_helper.get_semantic_signature(c.iri, lang)
                    c.relatedEntities = entities
                    c.relatedRelations = relations
                    c.numRelatedRelations = len(relations)
                    c.signatureOverlapScore = entity_linking_helper.computeSignatureOverlapScore(c.label, entities, sentence_tokens,
                                                                                     mention,
                                                                                     stopwords.words('english'))
                cdx = entity_linking_helper.rank_importance(candidates)

            results_all.append(mention_result)

    # Get candidates_per_sentence candidates for the whole sentence:
    results = []
    i = 0
    j = 0
    while len(results) < candidates_per_sentence:
        for mention_result in results_all:
            if (len(results) < candidates_per_sentence):
                mention = mention_result["mention"]
                try:
                    candidate = mention_result["mention_annotation"][i]
                    # results.append((candidate['babelSynsetID'],
                    #         main_sense,
                    #         str(counter + i),
                    #         support_word,
                    #         candidate['coherenceScore']))
                    ## TODO: set support_word = mention and score!
                    results.append((candidate["id"],
                                    candidate["label"],
                                    str(counter + j),
                                    mention,
                                    0.5
                                    ))
                    j += 1
                # for the candidate were no mentions found
                except:
                    continue
        i += 1
        # sometimes not enough candidates are available, then stop to prevent infinitve loop
        if (i > candidates_per_sentence):
            break

    return results, counter + len(results)
