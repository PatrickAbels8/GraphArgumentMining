import requests
import time
from json import JSONDecodeError
from query_kb import get_all_objects_sparql_ukp
from urllib.error import HTTPError
import urllib.parse, urllib.request
import networkx as nx
import os
import json

def get_property_infos(path):
    # load and return, if file exists
    if os.path.isfile(path+"property_frequencies.json"):
        with open(path+"property_frequencies.json") as f:
            return json.load(f)

    # if file does not exist, create it
    if os.path.isfile(path+"property_frequencies_raw.txt"):
        print("Create property frequency file from raw")
        result_dict = {}

        with open(path+"property_frequencies_raw.txt", encoding="utf-8") as f:
            for i, line in enumerate(f.readlines()):
                if i < 11:
                    continue
                if line.startswith("|-"):
                    continue

                line = line.split("||")

                try:
                    id = line[0].split("[[Property:")[1].split("|")[0]
                except IndexError as ie:
                    print(ie)
                    continue

                result_dict[id] = {
                    "label": line[1].strip(),
                    "description": line[2].strip(),
                    "aliases": line[3].strip(),
                    "data-type": line[4].strip(),
                    "count": int(line[5].strip().replace(",", ""))
                }

        # save
        with open(path+"property_frequencies.json", 'w') as outfile:
            json.dump(result_dict, outfile, indent=4, sort_keys=True)

        return result_dict

    return {}

def load_property_ids(path):
    id_list = []
    with open(path, "r") as f:
        for line in f:
            id_list.append(line.strip())
    return id_list

def generate_all_entity_combinations(annotations, remove_mirrored_paths=True):
    anno_dict = {}
    for object_id, object, subject_id, subject, _ in annotations:
        anno_dict[subject_id] = {"subject": subject, "object": object, "object_id": object_id}
    search_path_list = []
    for start in anno_dict.keys():
        for end in anno_dict.keys():
            if start != end:
                search_path_list.append((start, end))

    # remove duplicates
    if remove_mirrored_paths == True:
        final_list = []
        for start, end in search_path_list:
            if (start, end) not in final_list and (end, start) not in final_list:
                final_list.append((start, end))
        return final_list, anno_dict
    else:
        return search_path_list, anno_dict

def handle_urllib_request(query, url, method="POST"):
    query_counter = 0
    max_retries = 20
    delay = 10

    req = urllib.request.Request(url, data=query.encode("utf8"), method=method)
    while query_counter < max_retries:
        try:
            with urllib.request.urlopen(req, timeout = 60) as f:
                response = f.read()
                response = json.loads(response.decode("utf8"))
                query_counter = max_retries
        except HTTPError as he:
            query_counter += 1
            print("HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)
    return response

def handle_request(query):
    query_counter = 0
    max_retries = 20
    delay = 10

    while query_counter < max_retries:
        try:
            response = requests.get(query, timeout=60).json()
            break
        except HTTPError as he:
            query_counter += 1
            print("HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)
        except JSONDecodeError as jd:
            query_counter += 1
            print("JSONDecodeError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)

    if query_counter >= max_retries:
        print("Max retries reached, exit!")
        exit(1)

    return response

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def handle_request_babelnet(query, url, method="POST"):
    query_counter = 0
    max_retries = 20
    delay = 10

    while query_counter < max_retries:
        try:
            response = requests.get(url, params=query.encode("utf8"), timeout=60).json()
            break
        except HTTPError as he:
            query_counter += 1
            print("HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            print(str(he))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)
        except JSONDecodeError as jd:
            query_counter += 1
            print("JSONDecodeError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)

    if query_counter >= max_retries:
        print("Max retries reached, exit!")
        exit(1)

    return response

def get_object_query_func(kb_string):
    return {
        "wikidata_virtuoso": get_all_objects_sparql_ukp
    }[kb_string]

def get_shortest_paths(graph, start, end, consider_relation_directions=False):
    if consider_relation_directions == False:
        graph = graph.to_undirected()

    try:
        path = nx.shortest_path(graph, start, end)
    except nx.exception.NetworkXNoPath as e:
        return [], "", 0, 0, 0 # new
    path_list = []

    path_edges = list(zip(path, path[1:])) # new

    edgesinpath = list(zip(path[0:], path[1:]))
    path_len = len(edgesinpath)
    path_string = ""
    edge = "->" if consider_relation_directions == True else "-"
    for i, (u, v) in enumerate(edgesinpath):
        relations = "/".join(key + ";" + graph[u][v][key].get('pred_id', "") for key in graph[u][v].keys())
        s = graph.node[u]['name']
        o = graph.node[v]['name']
        path_list.append((s + ";" + u, relations, o + ";" + v))
        path_string += "[" + s + ";" + u + "]" + edge + "(" + relations + ")" + edge
        if i == path_len - 1:
            path_string += "[" + o + ";" + v + "]"

    return path_list, path_string, path_len, path_edges, path # new