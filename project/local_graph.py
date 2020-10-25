import time
from tqdm import tqdm
from helper import get_object_query_func
from entity_properties.pred_via_lda import props_per_entities
from pprint import pprint

import random
import networkx as nx
import matplotlib.pyplot as plt
from word_embeddings.sent_emb import preproc, sent_vec, cos_sim_sen



def add_to_subject_networkx(graph, subject_id, subject_name, predicates, predicate_ids, objects, object_ids, directions):
    for p, p_id, o, o_id, d in list(zip(predicates, predicate_ids, objects, object_ids, directions)):

        if d == "S->O":
            triple_exists = True if graph.has_edge(*(subject_id, o_id, p)) else False
        else:
            triple_exists = True if graph.has_edge(*(o_id, subject_id, p)) else False
 
        if triple_exists == False:
            if graph.has_node(subject_id) == False:
                graph.add_node(subject_id, name=subject_name)
            if graph.has_node(o_id) == False:
                graph.add_node(o_id, name=o)

            if d == "S->O":
                graph.add_edge(subject_id, o_id, key=p, pred_id=p_id)
            else:
                graph.add_edge(o_id, subject_id, key=p, pred_id=p_id)


def create(nx_graph, kb_string, annotations, kb_url, exclude_pred_by_name="", use_pred_by_name="",
           use_pred_by_id="", exclude_pred_by_id="", min_branches_per_node=15, max_nodes_per_hop=500, max_nodes=1200,
           printouts=True, consider_relation_directions=False, pred_infos=None, el_relation="WIKIFIERED",
           find_paths=True, retrieve_concept_names=True, max_graph_depth=-1, include_lda=False, include_wordvec=False, whole_sen = '',
           par_query=False):
    """
    Create local networkX graph
    :param annotations:
    :param entities_table: Entities table instance of neo4j
    :param entities: string entities for which to look for, e.g. "instance of; subclass of"
    :param no_hops: number of hops over the graph
    :param use_official_wikidata: use official wikidata or Daniil's
    """

    def init_nx_graph(annotations, graph):
        for annotation in annotations:
            object_id = annotation[0]  # related entity found for concept of sentence
            object_name = annotation[1]  # label for entity id
            subject_id = annotation[2]  # concept found in the input sentence
            subject_name = annotation[3]  # concept found in the input sentence
            if graph.has_node(subject_id) == False:
                graph.add_node(subject_id, name=subject_name)
            if graph.has_node(object_id) == False:
                graph.add_node(object_id, name=object_name)
            graph.add_edge(subject_id, object_id, key=el_relation)
            graph.add_edge(object_id, subject_id, key=el_relation)


    get_all_objects = get_object_query_func(kb_string)

    total_time = time.time()
    all_paths_exhausted = True
    if printouts == True:
        print("\n========== Create local graph ==========")

    init_nx_graph(annotations, nx_graph)
    if include_wordvec:
        w2v = preproc()
        sen_vec = sent_vec(w2v, whole_sen)

    if include_lda:
        entities = [an[1] for an in annotations]
        entity_preds = props_per_entities(cnt_topics = 5, entities=entities, max_props=150, count_threshold=40000)
        for (pred_id, pred_name) in entity_preds:
            use_pred_by_id.append(pred_id)

    # 2) DO BFS: search recursivly for x hops over the graph
    # 2.1) prevent creating a graph twice
    next_subject = []
    next_subject_ids = []
    for anno in annotations:
        if anno[0] not in next_subject_ids:
            next_subject.append(anno)
            next_subject_ids.append(anno[0])

    # 2.2) iterating over hops
    no_hops = 0
    visited = []
    while len(next_subject) > 0 and find_paths==True:
        if max_graph_depth != -1 and no_hops >= max_graph_depth:
            break
        no_hops += 1
        temp_next_subjects = []

        # calculate a limit of branching on each node by the max number of new nodes that are allowed each hop
        new_limit = int(max_nodes_per_hop/len(next_subject))
        if new_limit < min_branches_per_node:
            new_limit = min_branches_per_node

        if printouts == True:
            bar = "Processing hop " + str(no_hops) + " with " + str(len(next_subject)) + " next subjects and new limit of "+str(new_limit)+"...{l_bar}{bar}{r_bar}"
            next_subj_iterator = tqdm(next_subject, bar_format=bar)
        else:
            next_subj_iterator = next_subject
        # iterating over subjects left in queue
        for subject in next_subj_iterator:
            if subject[0] not in visited:
                visited.append(subject[0])

                temp_predicates, temp_p_ids, temp_objects, temp_o_ids, directions = [], [], [], [], []
            

                '''
                for every pred query via sparql for objects with sub+p_id and returns:
                    list: predicate names
                    list: predicate ids
                    list: object names
                    list: object ids
                    list: directions
                pred_infos are just sent all the way to ask for the label name when a pred_id is found
                '''
                if par_query:
                    tps, tpidss, tos, toidss, dss = get_all_objects(kb_url, subject[0], use_pred_by_id, limit=new_limit,
                                                     consider_relation_directions=consider_relation_directions, pred_infos=pred_infos,
                                                     retrieve_concept_names=retrieve_concept_names, par_query=par_query)
                    temp_predicates.extend(tps)
                    temp_p_ids.extend(tpidss)
                    temp_objects.extend(tos)
                    temp_o_ids.extend(toidss)
                    directions.extend(dss)
                else:
                    for pred in use_pred_by_id:
                        tp, tpids, \
                        to, toids, \
                        ds = get_all_objects(kb_url, subject[0], pred, limit=new_limit,
                                                     consider_relation_directions=consider_relation_directions, pred_infos=pred_infos,
                                                     retrieve_concept_names=retrieve_concept_names, par_query=par_query)
                        temp_predicates.extend(tp)
                        temp_p_ids.extend(tpids)
                        temp_objects.extend(to)
                        temp_o_ids.extend(toids)
                        directions.extend(ds)

                if consider_relation_directions == True and "O->S" in directions:
                    print("Error: O->S relation used although directed boolean is set to true.")


                if len(temp_predicates) > 0:
                    temp_list = list(zip(temp_o_ids, temp_objects, temp_p_ids, temp_predicates, directions))

                    # remove nodes with empty names
                    temp_list = [(o_id, o, p_id, p, d) for o_id, o, p_id, p, d in temp_list if o != "" and o != "None" and o != None]

                    # include only certain predicates (by name or id)
                    if len(use_pred_by_name) > 0 or len(use_pred_by_id) > 0:
                        temp_list = [(o_id, o, p_id, p, d) for o_id, o, p_id, p, d in temp_list if p in use_pred_by_name or p_id in use_pred_by_id]

                    # exclude certain predicates by name and id
                    if len(exclude_pred_by_name) > 0 or len(exclude_pred_by_id) > 0:
                        temp_list = [(o_id, o, p_id, p, d) for o_id, o, p_id, p, d in temp_list if p not in exclude_pred_by_name and p_id not in exclude_pred_by_id]

                    # next line only for o_id,o,p_id,p,d if cos_sim(o, subject[1])>threshold
                    if include_wordvec:
                        temp_next_subjects.extend([(o_id, o, subject[0]) for o_id, o, p_id, p, d in temp_list if (
                        o_id, o, subject[0]) not in temp_next_subjects and o_id not in visited and cos_sim_sen(w2v, sen_vec, o)])
                    else:
                        temp_next_subjects.extend([(o_id, o, subject[0]) for o_id, o, p_id, p, d in temp_list if (
                        o_id, o, subject[0]) not in temp_next_subjects and o_id not in visited])

                    if len(temp_list) > 0:
                        temp_o_ids, temp_objects, temp_p_ids, temp_predicates, directions = list(zip(*temp_list))

                        # add all found objects via the predicate to the subject in the graph
                        add_to_subject_networkx(nx_graph, subject[0], subject[1], temp_predicates, temp_p_ids,
                                       temp_objects, temp_o_ids, directions)

            # if the amount of visited nodes are bigger than max_nodes => stop
            if len(visited) >= max_nodes:
                all_paths_exhausted = False
                break


        next_subject = temp_next_subjects

        # if the amount of visited nodes + the amount of nodes that are next in the queue are bigger than max_nodes => stop
        if len(visited)+len(next_subject) >= max_nodes:
            all_paths_exhausted = False
            break

    total_time = time.time() - total_time
    if printouts == True:
        print("\n[local graph] Total time needed: " + '{0:.2f}'.format(float(total_time)/60) + " minutes")

    return total_time, no_hops, len(visited), all_paths_exhausted, len(use_pred_by_id)


