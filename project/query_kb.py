from wikidata import endpoint_access
import pandas as pd

def get_all_objects_sparql_ukp(db, subject_id, pred_id, limit=25, consider_relation_directions=False, pred_infos=None, par_query=False, **kwargs):
    assert consider_relation_directions == True, "Bidirectional queries have not been implemented for wikidata sparql endpoints!"

    if par_query:
        if len(pred_id) == 0:
            return [], [], [], [], []

    directions = []
    #wdt:P31 | wdt:P279

    if par_query:
        query = "PREFIX wd:<http://www.wikidata.org/entity/> PREFIX wdt:<http://www.wikidata.org/prop/direct/> SELECT DISTINCT  *  WHERE { wd:%s ?p ?o . FILTER (?p = wdt:%s "%(subject_id, pred_id[0])
        for pi in pred_id[1:]:
            query += " || ?p = wdt:%s "%pi
        query += " ) ?o rdfs:label ?label . FILTER (lang(?label) = 'en') . }"
    else:
        query = """
        PREFIX wd:<http://www.wikidata.org/entity/>
        PREFIX wdt:<http://www.wikidata.org/prop/direct/>
        
        SELECT DISTINCT  *  WHERE
        {     
        wd:%s wdt:%s ?o .
        ?o rdfs:label ?label .
        FILTER (lang(?label) = 'en') . 
        }
        """ % (subject_id, pred_id)

    if limit != -1 and limit > 0:
        query += "LIMIT " + str(limit)

    results = endpoint_access.query_wikidata(query, db)
    if len(results) == 0:  # no relations found
        return [], [], [], [], []
    results_df = pd.io.json.json_normalize(results)
    # results_df = pd.json_normalize(results)

    # retrieve objects and predicate URIs for the query
    temp_o_ids = results_df['o'].values.tolist()
    temp_objects = results_df['label'].values.tolist()
    directions = ["S->O"]*len(results_df)
    if par_query:
        temp_p_ids = results_df['p'].values.tolist()
    else:
        temp_p_ids = [pred_id] * len(results_df)

    # retrieve predicate labels for the predicate URIs
    temp_predicates = []
    for i, pred_uri in enumerate(
            temp_p_ids):  # https://www.mediawiki.org/wiki/Talk:Wikidata_Query_Service

        lbl = pred_infos.get(pred_uri, "N/A")
        if lbl != "N/A":
            lbl = lbl.get("label", "N/A")
        temp_predicates.append(lbl)

    return temp_predicates, temp_p_ids, temp_objects, temp_o_ids, directions

