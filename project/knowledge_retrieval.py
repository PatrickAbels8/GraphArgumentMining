import numpy as np
import time
from helper import generate_all_entity_combinations
from entity_linking import dbpedia_spotlight, wikifier
from helper import get_shortest_paths
import local_graph
# import local_graph_async as local_graph
import networkx as nx
import matplotlib.pyplot as plt
from external_knowledge.triple_retrieval import google_openie, add_triples_to_graph

def get_knowledge_for_sentence(sentence, topic, nx_graph, kb_string, kb_url, max_nodes=200, printouts=True, max_nodes_per_hop=500,
                               consider_relation_directions=False, remove_mirrored_paths=True, el_relation="el_relation",
                               exclude_pred_by_name=list(), pred_infos=None, use_pred_by_name="", max_graph_depth=-1, userkey="",
                               use_pred_by_id=list(), exclude_pred_by_id=list(), apply_page_rank_sq_threshold=True, use_wikifier=True, 
                               include_lda=False, include_wordvec=False, draw=False, par_query=False, include_openie=False):
    """
    Retrieves and adds knowledge for a sentence
    :param sentence: the sentence as string
    :param topic: the topic as string
    :param nx_graph: the networkx graph instance
    :param kb_string: knowledge base identifier
    :param kb_url: url to knowledge base
    :param consider_relation_directions: use or ignore directions of edges for the search query (only for neo4j)
    :param remove_mirrored_paths: removes the mirrored path search for concepts (e.g. if [(0,1),(1,0] -> [(0,1)]. Saves time
    if no directed search is used.
    :param use_pred_by_name: Uses only the predicates (name) given in the list for BFS
    :param exclude_pred_by_name: Excludes the predicates (name) given in the list for BFS
    :param use_pred_by_id: Uses only the predicates (wikidata ID) given in the list for BFS
    :param exclude_pred_by_id: Excludes the predicates (wikidata ID) given in the list for BFS
    :return: dict with metadata and paths to topic/between sentence; avg length of paths between conepts
    """

    total_time = time.time()

    # let wikifier find entities in the database for the tokens in the sentence and the topic
    if use_wikifier == True:
        sent_annotations, counter = wikifier(sentence, userkey, threshold=1, printouts=printouts, limit=5,
                                             apply_page_rank_sq_threshold=apply_page_rank_sq_threshold,
                                             retrieve_concept_names=True)  # get annotations found for the given sentence in form of [(entity_id, label), ...]
        topic_annotations, counter = wikifier(topic, userkey, threshold=1, counter=counter, topic=True, printouts=printouts,
                                              limit=5, apply_page_rank_sq_threshold=apply_page_rank_sq_threshold,
                                              retrieve_concept_names=True)
    else:
        sent_annotations, counter = dbpedia_spotlight(sentence, printouts=printouts, limit=5,
                                                  apply_page_rank_sq_threshold=apply_page_rank_sq_threshold,
                                                  knowledge = "wikidata_truthy")  # get annotations found for the given sentence in form of [(entity_id, label), ...]
        topic_annotations, counter = dbpedia_spotlight(topic, counter=counter, topic=True, printouts=printouts,
                                                   limit=5, apply_page_rank_sq_threshold=apply_page_rank_sq_threshold,
                                                   knowledge = "wikidata_truthy")

    topic_ids = [id for _, _, id, _, _ in topic_annotations]
    all_annotations = sent_annotations.copy()
    all_annotations.extend(topic_annotations)
    all_annotations = list(set(all_annotations))

    # generate all combinations between entities found for later path queries
    combinations, anno_dict = generate_all_entity_combinations(all_annotations, remove_mirrored_paths=remove_mirrored_paths)

    if printouts == True:
        print("\n========== Try to find paths between the following nodes ==========")
        for combination in combinations:
            print(anno_dict[combination[0]]['subject'] + "<=>" + anno_dict[combination[1]]['subject'])

    # create local graph from BFS
    _, no_hops, no_visited_nodes, all_paths_exhausted, no_preds = local_graph.create(nx_graph, kb_string, all_annotations, kb_url,
                                                                           use_pred_by_name=use_pred_by_name,
                                                                           exclude_pred_by_name=exclude_pred_by_name,
                                                                           use_pred_by_id=use_pred_by_id,
                                                                           exclude_pred_by_id=exclude_pred_by_id,
                                                                           max_nodes_per_hop=max_nodes_per_hop,
                                                                          max_nodes=max_nodes, printouts=printouts,
                                                                           consider_relation_directions=consider_relation_directions,
                                                                           el_relation=el_relation,
                                                                           find_paths=True, pred_infos=pred_infos,
                                                                           retrieve_concept_names=True,
                                                                           max_graph_depth=max_graph_depth,
                                                                           include_lda=include_lda,
                                                                           include_wordvec=include_wordvec,
                                                                           whole_sen=sentence,
                                                                           par_query=par_query)

    print('##### graph nodes before:', len(list(nx_graph.nodes.data())))
    
    # add up local_graph and google_graph (new)
    if include_openie:
      if printouts == True:
        print('\n========== Create external graph ==========')
      google_triples = google_openie(query=topic, max_triples=500)
      google_nodes = add_triples_to_graph(graph=nx_graph, triples=google_triples)
      if printouts:
        print('##### found triples:', len(google_triples))
        print('##### graph nodes after:', len(list(nx_graph.nodes.data())))


    # create results dictionary
    final_results = {
        'total_processing_time_seconds': '{0:.0f}'.format(time.time() - total_time),
        'total_visited_nodes':no_visited_nodes,
        'total_hops': no_hops,
        'total_sent_annotations':len(all_annotations)-len(topic_annotations),
        'total_topic_annotations':len(topic_annotations),
        'paths_to_topic': [],
        'topic_concepts': ["["+sup_w_label+";"+str(sup_w_id)+"]->("+el_relation+";)->["+c_label+";"+str(c_id)+";"+str(conf)+"]" for c_id, c_label, sup_w_id, sup_w_label, conf in topic_annotations],
        'sent_concepts': ["["+sup_w_label+";"+str(sup_w_id)+"]->("+el_relation+";)->["+c_label+";"+str(c_id)+";"+str(conf)+"]" for c_id, c_label, sup_w_id, sup_w_label, conf in sent_annotations],
        'total_paths_to_topic': 0,
        'paths_within_sent': [],
        'total_paths_within_sent': 0,
        'all_paths_exhausted': all_paths_exhausted,
        'total_properties': no_preds
    }

    print("\n========== Found paths between the nodes ==========")

    if draw:
      pos = nx.spring_layout(nx_graph) # new
      nx.draw_networkx_nodes(nx_graph, pos, node_color='k', node_size=1) # new
      nx.draw_networkx_edges(nx_graph, pos) # new
      # nx.draw_networkx_edge_labels(nx_graph, pos) # new
      nx.draw_networkx_labels(nx_graph, pos, font_size=6) # new

      if include_openie:
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=google_nodes, node_color='r', node_size=2) # new


    avg_path_len = []
    for start, end in combinations:

        if start in topic_ids and end in topic_ids: # dont search paths between topic entities
            continue

        # get shortest path
        path_list, path_string, path_len, path_edges, path = get_shortest_paths(nx_graph, start, end, # new
                                                              consider_relation_directions=consider_relation_directions)


      
        if path_len > 0:
            avg_path_len.append(path_len)
            if draw:
              nx.draw_networkx_nodes(nx_graph,pos,nodelist=path,node_color='r', node_size=10) # new
              nx.draw_networkx_edges(nx_graph,pos,edgelist=path_edges,edge_color='r',width=2) # new

        # save path (if exists) in results file
        if len(path_string) > 0:
            # if a topic is involved in the path, save it to a separate path list
            if start in topic_ids or end in topic_ids:
                final_results['paths_to_topic'].append(path_string)
            else:
                final_results['paths_within_sent'].append(path_string)

            # print path if exists
            print(anno_dict[start]['subject'] + "=>" + anno_dict[end]['subject'] + ":\t" + path_string)
    if draw:
      plt.axis('off')
      plt.show() # new

    # calculate avg path lengths between concepts
    avg_path_len = np.average(avg_path_len) if len(avg_path_len) > 0 else 0

    # add the number of paths between entities and entities/topics to the result
    final_results['total_paths_to_topic'] = len(final_results['paths_to_topic'])
    final_results['total_paths_within_sent'] = len(final_results['paths_within_sent'])

    return final_results, avg_path_len