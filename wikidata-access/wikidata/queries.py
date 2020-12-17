import json
import urllib
from collections import defaultdict

from wikidata import endpoint_access, scheme

ENTITY_VAR = "?evar"
LABEL_VAR = "?label"

sparql_inference_clause = """
        DEFINE input:inference 'instances'
        """

sparql_transitive_option = "option (transitive,t_no_cycles, t_min (1), t_max(5), t_step ('step_no') as ?step)"

sparql_prefix = """
        PREFIX g:<http://wikidata.org/>
        PREFIX e:<http://www.wikidata.org/entity/>
        PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
        PREFIX base:<http://www.wikidata.org/ontology#>
        PREFIX schema:<http://schema.org/>
        """

sparql_select = """
        SELECT DISTINCT {queryvariables} WHERE
        """

sparql_ask = """
        ASK WHERE
        """

sparql_close_order = " ORDER BY {}"
sparql_close = " LIMIT {}"

sparql_get_entity_by_label = """
        {{
              {{VALUES ?labelpredicate {{rdfs:label skos:altLabel}}
              GRAPH g:terms {{ ?evar ?labelpredicate "{entitylabel}"@en  }}
              }}
        }}
        FILTER EXISTS {{ GRAPH g:statements {{ ?evar ?p ?v }} }}
        FILTER ( NOT EXISTS {{GRAPH g:instances {{?evar rdf:type e:Q17442446}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?evar rdf:type e:Q15474042}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?evar rdf:type e:Q18616576}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?evar rdf:type e:Q5707594}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?evar rdf:type e:Q427626}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?evar rdf:type e:Q16521}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?evar rdf:type e:Q11173}} }}                               
            )
        """

sparql_get_year_from_entity = """
        {{
        VALUES ?evar {{ {entityids} }}
        GRAPH g:statements {{ ?evar base:time ?et. BIND (YEAR(?et) AS ?label) }}
        }}
        """

sparql_get_entity_labels = """
        {{
        VALUES ?evar {{ {entityids} }}
        VALUES ?labelpredicate {{rdfs:label skos:altLabel}}
        GRAPH g:terms {{ ?evar ?labelpredicate ?label }}        
        }}
        """

sparql_get_entity_labels_all = """
        {{
        VALUES ?evar {{ {entityids} }}
        GRAPH g:terms {{ ?evar rdfs:label ?label }}      
        FILTER ( lang(?label) = "en" )  
        }}
        """


sparql_get_main_entity_label = """
        {{
        GRAPH g:terms {{ {entityid} rdfs:label ?label }}
        FILTER ( lang(?label) = "en" )
        }}
        """


sparql_map_f_id = """
        {{
          GRAPH g:statements {{ ?evar e:P646s/e:P646v "{otherkbid}" }}
        }}
        """


sparql_map_wikipedia_id = """
        {{
            GRAPH g:sitelinks {{ <{otherkbid}> schema:about ?evar }}
        }}
        """

sparql_get_all_predicates_or_objects = """
            ?evar ?p ?o .
            GRAPH g:terms { ?o rdfs:label ?label }
            FILTER ( lang(?label) = "en" )
        """

def query_base(limit=None):
    query = sparql_prefix + sparql_select
    query += "{{ {mainquery} }}"
    if limit:
        query += sparql_close.format(limit)
    return query


def query_entity_base(sparql_template, entity=None, limit=None, queryvariables=None):
    query = query_base(limit)
    if entity:
        sparql_template = sparql_template.replace(ENTITY_VAR, f"e:{entity}")
    query = query.format(mainquery=sparql_template, queryvariables=queryvariables)
    return query


def query_get_labels_for_entities(entities, limit_per_entity=10, all_labels=False):
    """
    Construct a WikiData query to retrieve entity labels for the given list of entity ids.

    :param entities: entity kbIDs
    :param limit_per_entity: limit on the result list size (multiplied with the size of the entity list)
    :return: a WikiData query
    >>> endpoint_access.query_wikidata(query_get_labels_for_entities(["Q36", "Q76"]))  # doctest: +ELLIPSIS
    [{'evar': 'Q36', 'label': 'Poland'}, ..., {'evar': 'Q76', 'label': 'Obama'}, ...]
    """
    if all(e[0] not in 'qQ' or '-' in e for e in entities):
        query = sparql_get_year_from_entity
    else:
        entities = [e for e in entities if '-' not in e and e[0] in 'pqPQ']
        query = sparql_get_entity_labels if all_labels == False else sparql_get_entity_labels_all
    base = query_base(limit=limit_per_entity * len(entities))
    query = base.format(queryvariables=" ".join([ENTITY_VAR, LABEL_VAR]),
                        mainquery=query.format(entityids=" ".join(["e:" + entity for entity in entities]),))
    return query


def get_labels_for_entities(entities):
    """
    Label the given set of variables with all labels available for them in the knowledge base.

    :param entities: a list of entity ids.
    :return: a dictionary mapping entity id to a list of labels
    >>> dict(get_labels_for_entities(["Q76", "Q188984", "Q194339"])) == \
    {'Q188984': {'NY Rangers', 'Blue-Shirts', 'Blue Shirts', 'Broadway Blueshirts', 'New York Rangers', 'NYR'}, 'Q194339': {'Bahamian dollar', 'Bahama-Dollar', 'B$', 'Bahamas-Dollar'}, 'Q76': {'Barack Obama', 'Barack Hussein Obama II', 'Barack H. Obama', 'Barack Obama II', 'Barack Hussein Obama, Jr.', 'Barack Hussein Obama', 'Obama'}}
    True
    >>> dict(get_labels_for_entities(["VTfb0eeb812ca69194eaaa87efa0c6d51d"]))
    {'VTfb0eeb812ca69194eaaa87efa0c6d51d': {'1972'}}
    """
    results = endpoint_access.query_wikidata(query_get_labels_for_entities(entities))
    entity_name = ENTITY_VAR[1:]
    label_name = LABEL_VAR[1:]
    if len(results) > 0:
        retrieved_labels = defaultdict(set)
        for result in results:
            entity_id = result[entity_name].replace(scheme.WIKIDATA_ENTITY_PREFIX, "")
            retrieved_labels[entity_id].add(result[label_name])
        return retrieved_labels
    return {}


def query_get_main_entity_label(entity):
    """
    Construct a WikiData query to retrieve the main entity label for the given entity id.

    :param entity: entity kbID
    :return: a WikiData query
    >>> endpoint_access.query_wikidata(query_get_main_entity_label("Q36"))
    [{'label': 'Poland'}]
    """
    query = query_base().format(queryvariables="?label",
                                mainquery=sparql_get_main_entity_label.format(entityid=f"e:{entity}"))
    return query


def get_main_entity_label(entity):
    """
    Retrieve the main label of the given entity. None is returned if no label could be found.

    :param entity: entity KB ID
    :return: entity label as a string
    >>> get_main_entity_label("Q12143")
    'time zone'
    """
    results = endpoint_access.query_wikidata(query_get_main_entity_label(entity))
    if results and 'label' in results[0]:
        return results[0]['label']
    return None


def query_get_entity_by_label(label):
    """
    A method to look up a WikiData entity by a label. Only exact matches are returned.

    :param label: entity label
    :param limit: limit on the result list size
    :return: a query that can be executed against WikiData
    >>> endpoint_access.query_wikidata(query_get_entity_by_label("Barack Obama"))
    [{'evar': 'Q76'}]
    >>> 'Q8027' in {r['evar'] for r in endpoint_access.query_wikidata(query_get_entity_by_label("Martin Luther King"))}
    True
    """
    base = query_base(limit=10)
    query = base.format(queryvariables=ENTITY_VAR,
                        mainquery=sparql_get_entity_by_label.format(entitylabel=label))
    query = sparql_inference_clause + query
    return query


def query_map_freebase_id(f_id):
    """
    Map a Freebase id to a Wikidata entity

    :param f_id: Freebase id
    :return: a WikiData query
    >>> endpoint_access.query_wikidata(query_map_freebase_id("/m/0d3k14"))
    [{'evar': 'Q9696'}]
    """
    base = query_base(limit=1)
    query = base.format(queryvariables=ENTITY_VAR,
                        mainquery=sparql_map_f_id.format(otherkbid=f_id))
    return query


def query_map_wikipedia_id(wikipedia_article_id):
    """
    Map a Wikipedia id to a Wikidata entity

    :param wikipedia_article_id: Freebase id
    :return: a WikiData query
    >>> endpoint_access.query_wikidata(query_map_wikipedia_id("John_F._Kennedy"))
    [{'evar': 'Q9696'}]
    """
    wikipedia_article_id = wikipedia_article_id.replace(scheme.WIKIPEDIA_PREFIX, "")
    wikipedia_article_id = urllib.parse.quote(wikipedia_article_id, safe="/:")
    wikipedia_article_id = scheme.WIKIPEDIA_PREFIX + wikipedia_article_id

    base = query_base(limit=1)
    query = base.format(queryvariables=ENTITY_VAR,
                        mainquery=sparql_map_wikipedia_id.format(otherkbid=wikipedia_article_id))

    return query


def map_f_id(f_id):
    """
    Map the given Freebase id to a Wikidata id

    :param f_id: Freebase id as a string
    :return: Wikidata kbID
    >>> map_f_id('/m/0drf_')
    'Q51789'
    """
    f_id = f_id.replace(".", "/")
    if not f_id.startswith("/"):
        f_id = "/" + f_id
    results = endpoint_access.query_wikidata(query_map_freebase_id(f_id))
    if results and ENTITY_VAR[1:] in results[0]:
        return results[0][ENTITY_VAR[1:]]
    return None


def map_wikipedia_id(wikipedia_article_id):
    """
    Map the given Wikipedia article URL (id) to a Wikidata id

    :param wikipedia_article_id: Wikipedia id as a string
    :return: Wikidata kbID
    >>> map_wikipedia_id("PIAS_Entertainment_Group")
    'Q7119302'
    >>> map_wikipedia_id("Swimming_(sport)")
    'Q31920'
    >>> map_wikipedia_id("José_Reyes_(shortstop)")
    'Q220096'
    >>> map_wikipedia_id("The_Twilight_Saga:_New_Moon")
    'Q116928'
    >>> map_wikipedia_id("betty_ford_center")
    'Q850360'
    >>> map_wikipedia_id("1976_democratic_national_convention")
    'Q16152917'
    """
    results = endpoint_access.query_wikidata(query_map_wikipedia_id(wikipedia_article_id))
    evar_name = ENTITY_VAR[1:]
    if results and evar_name in results[0]:
        return results[0][evar_name]
    response = urllib.request.urlopen("https://en.wikipedia.org/w/api.php?action=query&redirects=1&format=json&prop=info&inprop=url&titles=" +
                                      urllib.parse.quote(wikipedia_article_id))
    encoding = response.info().get_content_charset("utf-8")
    json_response = json.loads(response.read().decode(encoding))
    if 'query' in json_response and 'pages' in json_response['query']:
        json_response = list(json_response['query']['pages'].items())
        k, value = json_response[0]
        if k != -1 and 'canonicalurl' in value:
            canonical_article_url = urllib.parse.unquote(value['canonicalurl'])
            results = endpoint_access.query_wikidata(query_map_wikipedia_id(canonical_article_url))
            if results and evar_name in results[0]:
                return results[0][evar_name]

    capitalized = "_".join([token.title() for token in wikipedia_article_id.split("_")])
    if capitalized != wikipedia_article_id:
        return map_wikipedia_id(capitalized)
    return None


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
