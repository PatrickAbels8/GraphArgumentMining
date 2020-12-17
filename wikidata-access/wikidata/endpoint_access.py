import atexit
import pickle
import logging
import os

from SPARQLWrapper import SPARQLWrapper, JSON

from wikidata import scheme



#logger = wdaccess_p['logger']
#logger.setLevel(logging.INFO)



'''
FILTER_RELATION_CLASSES = "qr"

query_cache = {}
cached_counter = 0
query_counter = 1

cache_location = os.path.dirname(__file__)
if wdaccess_p['use.cache'] and os.path.isfile(cache_location + "/.wdacess.cache"):
    try:
        with open(cache_location + "/.wdacess.cache") as f:
            query_cache = pickle.load(f)
        logger.info("Query cache loaded. Size: {}".format(len(query_cache)))
    except Exception as ex:
        logger.error("Query cache exists, but can't be loaded. {}".format(ex))


def clear_cache():
    global query_cache
    query_cache = {}


def dump_cache():
    if wdaccess_p['use.cache']:
        logger.info("Cached query ratio: {}.".format(cached_counter / query_counter))
    if query_cache:
        logger.info("Dump query cache.")
        with open(cache_location + "/.wdacess.cache", "w") as out:
            pickle.dump(query_cache, out)


atexit.register(dump_cache)


def filter_relations(results, b='p', freq_threshold=0):
    """
    Takes results of a SPARQL query and filters out all rows that contain blacklisted relations.

    :param results: results of a SPARQL query
    :param b: the key of the relation value in the results dictionary
    :return: filtered results
    >>> filter_relations([{"p":"http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "e2":"http://www.wikidata.org/ontology#Item"}, {"p":"http://www.wikidata.org/entity/P1429s", "e2":"http://www.wikidata.org/entity/Q76S69dc8e7d-4666-633e-0631-05ad295c891b"}])
    []
    """
    results = [r for r in results if b not in r or
               (r[b][:-1] in scheme.content_properties and r[b][-1] not in FILTER_RELATION_CLASSES)
               ]
    results = [r for r in results if b not in r or scheme.property2label.get(r[b][:-1], {}).get('freq') > freq_threshold]
    return results
'''
query_cache = {}
cached_counter = 0
query_counter = 1
def query_wikidata(query, db, prefix=scheme.WIKIDATA_ENTITY_PREFIX, use_cache=-1, timeout=-1):
    """
    Execute the following query against WikiData
    :param query: SPARQL query to execute
    :param prefix: if supplied, then each returned URI should have the given prefix. The prefix is stripped
    :param use_cache: set to 0 or 1 to override the global setting
    :param timeout: set to a value large than 0 to override the global setting
    :return: a list of dictionaries that represent the queried bindings
    """

    wdaccess_p = {
        'backend': db,
        'timeout': 20,
        'global_result_limit': 1000,
        'logger': logging.getLogger(__name__),
        'use.cache': False,
        'mode': "quality"  # options: precision, fast
    }

    def get_backend(backend_url):
        global sparql
        sparql = SPARQLWrapper(backend_url)
        sparql.setReturnFormat(JSON)
        sparql.setMethod("GET")
        sparql.setTimeout(wdaccess_p.get('timeout', 40))
        return sparql

    sparql = get_backend(wdaccess_p.get('backend', "http://knowledgebase:8890/sparql"))
    #GLOBAL_RESULT_LIMIT = wdaccess_p['global_result_limit']

    use_cache = (wdaccess_p['use.cache'] and use_cache != 0) or use_cache == 1
    global query_counter, cached_counter, query_cache
    query_counter += 1
    if use_cache and query in query_cache:
        cached_counter += 1
        return query_cache[query]
    if timeout > 0:
        sparql.setTimeout(timeout)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except Exception as inst:
        #print(inst)
        return []
    # Change the timeout back to the default
    if timeout > 0:
        sparql.setTimeout(wdaccess_p.get('timeout', 40))
    if "results" in results and len(results["results"]["bindings"]) > 0:
        results = results["results"]["bindings"]
        #print(f"Results bindings: {results[0].keys()}")
        if prefix:
            results = [r for r in results if all(not r[b]['value'].startswith("http://") or r[b]['value'].startswith(prefix) for b in r)]
        results = [{b: (r[b]['value'].replace(prefix, "") if prefix else r[b]['value']) for b in r} for r in results]
        if use_cache:
            query_cache[query] = results
        return results
    elif "boolean" in results:
        return results['boolean']
    else:
        #print(results)
        return []


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
