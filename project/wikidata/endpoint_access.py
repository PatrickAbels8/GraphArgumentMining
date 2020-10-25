
import logging

from SPARQLWrapper import SPARQLWrapper, JSON

from wikidata import scheme

query_cache = {}
cached_counter = 0
query_counter = 1
def query_wikidata(query, db, e_prefix=scheme.WIKIDATA_ENTITY_PREFIX, p_prefix=scheme.WIKIDATA_PROPERTY_PREFIX, use_cache=-1, timeout=-1):
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

    sparql = get_backend(wdaccess_p.get('backend', "http://knowledge-graph:8890/sparql"))
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
        # print(inst)
        return []
    # Change the timeout back to the default
    if timeout > 0:
        sparql.setTimeout(wdaccess_p.get('timeout', 40))
    if "results" in results and len(results["results"]["bindings"]) > 0:
        results = results["results"]["bindings"]
        #print(f"Results bindings: {results[0].keys()}")
        if e_prefix and p_prefix: #new
            results = [r for r in results if all(not r[b]['value'].startswith("http://") or r[b]['value'].startswith(e_prefix) or r[b]['value'].startswith(p_prefix) for b in r)]
        results = [{b: (r[b]['value'].replace(e_prefix, "").replace(p_prefix, "") if e_prefix and p_prefix else r[b]['value']) for b in r} for r in results]
        if use_cache:
            query_cache[query] = results
        return results
    elif "boolean" in results:
        return results['boolean']
    else:
        return []


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
