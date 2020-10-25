from wikidata.scheme import content_properties, frequent_properties
from flask import Flask, send_from_directory, request, render_template, redirect, url_for
import json
from knowledge_retrieval import get_knowledge_for_sentence
import logging.config
import os, errno
import helper
import networkx as nx

from time import sleep, time
import multiprocessing
from datetime import datetime
import random

import argparse


def extract_evidence(json_post, include_lda=True, include_wordvec=False, draw=False, par_query=True, include_openie=False):
  try:
      # read input json
      # json_post = request.get_json()

      if isinstance(json_post, dict):
          # default values
          use_pred_by_name = [] #["INSTANCE_OF", "SUBCLASS_OF"]
          exclude_pred_by_name = ["FREEBASE_ID", "IMAGE", "QUORA_TOPIC_ID", "DESCRIBED_BY_SOURCE"]  #
          exclude_pred_by_id = helper.load_property_ids("resources/property_blacklist.txt")
          apply_page_rank_sq_threshold = True # wikifier property
          consider_relation_directions = True  # builds the local graph in a directed way (only S->O) and uses shortest path only in a direct way
          el_relation = "WIKIFIERED" # default entity relation between token in sentence/topic and first entity found in KB
          wikifier_userkey = "zlqsluqpacmwysgzyonpvqkmvmixiu" 

          # input args
          topic = json_post['topic']
          sent = json_post['sent']
          use_wikifier = json_post.get('use_wikifier', True)
          use_official_endpoint = json_post.get("use_official_endpoint", True)
          use_pred_by_id = json_post.get("use_pred_by_id", ["P279", "P31", "P361", "P460"]) # P279 (subclassof), P31 (instanceof), P361 (partof), P460 (saidtobethesameas)
          max_graph_depth = json_post.get("max_graph_depth", 3)
          max_nodes = json_post.get("max_nodes", 600) # max number of nodes visited before stop
          max_nodes_per_hop = json_post.get("max_nodes_per_hop", 200) # max nodes to visit before stop
          if "CONTENT_PROPERTIES" in use_pred_by_id:
              use_pred_by_id = [el for el in use_pred_by_id if el != "CONTENT_PROPERTIES"]
              use_pred_by_id.extend(content_properties)
          if "FREQUENT_PROPERTIES" in use_pred_by_id:
              use_pred_by_id = [el for el in use_pred_by_id if el != "FREQUENT_PROPERTIES"]
              use_pred_by_id.extend(frequent_properties)

          # dic with preds, alias, desxriptions, freqs
          pred_infos = helper.get_property_infos("wikidata/resources/wikidata_30-11-18/")

          # get db connections
          kb_name = "wikidata_virtuoso"
          if use_official_endpoint == True:
              kb_url = "https://query.wikidata.org/sparql"
          else:
              kb_url = "http://knowledge-graph:8890/sparql"
          nx_graph = nx.MultiDiGraph()

          results, avg_path_len = get_knowledge_for_sentence(sent, topic.replace("_", " "), nx_graph, kb_name, kb_url,
                                                             max_nodes=max_nodes,
                                                             max_nodes_per_hop=max_nodes_per_hop,
                                                             el_relation=el_relation,
                                                             use_pred_by_name=use_pred_by_name,
                                                             consider_relation_directions=consider_relation_directions,
                                                             remove_mirrored_paths=False,
                                                             use_wikifier=use_wikifier,
                                                             printouts=True,
                                                             exclude_pred_by_name=exclude_pred_by_name,
                                                             use_pred_by_id=use_pred_by_id,
                                                             exclude_pred_by_id=exclude_pred_by_id, pred_infos=pred_infos,
                                                             apply_page_rank_sq_threshold=apply_page_rank_sq_threshold,
                                                             max_graph_depth=max_graph_depth,
                                                             userkey=wikifier_userkey,
                                                             include_lda=include_lda,
                                                             include_wordvec=include_wordvec,
                                                             draw=draw,
                                                             par_query=par_query,
                                                             include_openie=include_openie)
          results["topic"] = topic
          results["sent"] = sent
          # print(json.dumps(results, sort_keys=True, indent=4))
          return(json.dumps(results, sort_keys=True, indent=4))

  except Exception as e:
      print(str(e))
      return json.dumps({"error": "Execution error. Please contact an administrator."})

def get_cases(path, count=1, start=0):
  with open(path, 'r', encoding='utf8') as f:
    data = f.readlines()
  data = data[-len(data)+1:] # cut titles
  data = data[start:count+start] # only tkae first count from start on
  args = [(d.split('\t')[0], d.split('\t')[3], d.split('\t')[4], d.split('\t')[5]) for d in data] # collect arguments (topic + hash + sentence + label)
  return args

def run_and_save(topic, hashstr, sentence, label, max_depth, include_lda, include_wordvec, par_query, include_openie, output_path):
  f_json = extract_evidence({"topic": topic, "sent": sentence, "max_graph_depth": max_depth}, include_lda=include_lda, include_wordvec=include_wordvec, par_query=par_query, include_openie=include_openie)
  f = json.loads(f_json)
  f['label'] = 'Argument_for' if 'for' in label else 'Argument_against' if 'against' in label else 'NoArgument'
  f['hash'] = hashstr
  f_print_json = json.dumps(f)
  with open(output_path, 'a') as ff:
    ff.write(f_print_json)
    ff.write('\n')

def create_classifier_data(topi, output_path, classifier_path):
  with open(output_path, 'r') as f:
    samples = f.read()

  samples = samples.split('}')
  ret_samples = []

  all_paths_exhausted = True
  avg_no_hops = 0
  avg_no_paths_between_entities = 0
  avg_no_paths_to_topic = 0
  avg_no_visited_nodes = 0
  avg_path_len = 0
  avg_processing_time_seconds = 0
  total_processing_time_seconds = 0
  total_samples = 0
  total_samples_with_paths = 0

  for s in samples:
    sample = s.replace('\n', '')+'}'
    try:
      sample_dict = json.loads(sample)
    except:
      continue
    if 'error' not in sample_dict:
      avg_pathlen = 0
      for p in sample_dict['paths_within_sent']:
        avg_pathlen += p.count('[')
      for p in sample_dict['paths_to_topic']:
        avg_pathlen += p.count('[')
      try:
        avg_pathlen = avg_pathlen/(len(sample_dict['paths_within_sent'])+len(sample_dict['paths_to_topic']))
      except ZeroDivisionError:
        avg_pathlen = 0

      sample_dict['avg_path_len'] = avg_pathlen
      sample_dict['set'] = random.choice(['test', 'train', 'val'])
      sample_dict['no_hops'] = sample_dict.pop('total_hops')
      sample_dict['no_paths_between_entities'] = sample_dict.pop('total_paths_within_sent')
      sample_dict['no_paths_to_topic'] = sample_dict.pop('total_paths_to_topic')
      sample_dict['no_sent_annotations'] = sample_dict.pop('total_sent_annotations')
      sample_dict['no_topic_annotations'] = sample_dict.pop('total_topic_annotations')
      sample_dict['no_visited_nodes'] = sample_dict.pop('total_visited_nodes')
      sample_dict['paths_between_entities'] = sample_dict.pop('paths_within_sent')
      sample_dict['sentence'] = sample_dict.pop('sent')

      all_paths_exhausted = all_paths_exhausted and sample_dict['all_paths_exhausted']
      avg_no_hops += sample_dict['no_hops']
      avg_no_paths_between_entities += sample_dict['no_paths_between_entities']
      avg_no_paths_to_topic += sample_dict['no_paths_to_topic']
      avg_no_visited_nodes += sample_dict['no_visited_nodes']
      avg_path_len += sample_dict['avg_path_len']
      total_processing_time_seconds += float(sample_dict['total_processing_time_seconds'])
      total_samples += 1
      if sample_dict['no_paths_between_entities']+sample_dict['no_paths_to_topic'] > 0:
        total_samples_with_paths += 1

      ret_samples.append(sample_dict)

  out_dict = {
    'metadata': {
      'all_paths_exhausted' : all_paths_exhausted,
      'avg_no_hops' : avg_no_hops/total_samples,
      'avg_no_paths_between_entities' : avg_no_paths_between_entities/total_samples,
      'avg_no_paths_to_topic' : avg_no_paths_to_topic/total_samples,
      'avg_no_visited_nodes' : avg_no_visited_nodes/total_samples,
      'avg_path_len' : avg_path_len/total_samples,
      'avg_processing_time_seconds' : total_processing_time_seconds/total_samples,
      'total_processing_time_seconds' : total_processing_time_seconds,
      'total_samples' : total_samples,
      'total_samples_with_paths' : total_samples_with_paths,
      'creation_date': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
      'topic': topic
    },
    'samples': ret_samples
  }


  with open(classifier_path, 'w') as f:
    f.write(json.dumps(out_dict, sort_keys=True, indent=4))

def main():

  # process arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', defalt=None, type=str, required=True, help='No data_dir found.')
  parser.add_argument('--topic', defalt=None, type=str, required=True, help='No topic found.')
  parser.add_argument('--num_cases', defalt=None, type=str, required=True, help='No num_cases found.')
  parser.add_argument('--depth', defalt=None, type=str, required=True, help='No depth found.')
  parser.add_argument('--include_lda', defalt=None, type=str, required=True, help='No include_lda found.')
  parser.add_argument('--include_wordvec', defalt=None, type=str, required=True, help='No include_wordvec found.')
  parser.add_argument('--par_query', defalt=None, type=str, required=True, help='No par_query found.')
  parser.add_argument('--include_openie', defalt=None, type=str, required=True, help='No include_openie found.')
  args = parser.parse_args()

  # load data
  os.rename('data//abortion.tsv', 'testing//abortion.tsv')
  os.rename('data//cloning.tsv', 'testing//cloning.tsv')
  os.rename('data//death_penalty.tsv', 'testing//death_penalty.tsv')
  os.rename('data//gun_control.tsv', 'testing//gun_control.tsv')
  os.rename('data//marijuana_legalization.tsv', 'testing//marijuana_legalization.tsv')
  os.rename('data//minimum_wage.tsv', 'testing//minimum_wage.tsv')
  os.rename('data//nuclear_energy.tsv', 'testing//nuclear_energy.tsv')
  os.rename('data//school_uniforms.tsv', 'testing//school_uniforms.tsv')
  os.rename('data//glove.6B.100d.txt', 'word_embeddings//glove.6B//glove.6B.100d.txt')
  os.rename('data//glove.6B.200d.txt', 'word_embeddings//glove.6B//glove.6B.200d.txt')
  os.rename('data//glove.6B.300d.txt', 'word_embeddings//glove.6B//glove.6B.300d.txt')
  os.rename('data//glove.6B.50d.txt', 'word_embeddings//glove.6B//glove.6B.50d.txt')
  os.rename('data//property_frequencies.json', 'entity_properties//property_frequencies.json')
  os.rename('data//properties_with_labels.txt', 'resources//properties_with_labels.txt')
  os.rename('data//wikidata_30-11-18//property_frequencies.json', 'wikidata//resources//wikidata_30-11-18//property_frequencies.json')
  os.rename('data//wikidata_30-11-18//property_frequencies_raw.txt', 'wikidata//resources//wikidata_30-11-18//property_frequencies_raw.txt')
  os.rename('data//wikidata_ukp//properties_with_labels.txt', 'wikidata//resources//wikidata_ukp//properties_with_labels.txt')
  os.rename('data//wikidata_ukp//property_blacklist.txt', 'wikidata//resources//wikidata_ukp//property_blacklist.txt')

  # create paths
  path_to_cases = 'testing//'+args.topic+'.tsv' # can change to any .tsv in /testing
  output_path = 'results//raw_result_'+args.topic+'_nx.json'
  classifier_path = output_path.replace('raw_', '')

  # load cases, run model and translate to json for classifier
  cases = get_cases(path_to_cases, args.num_cases)
  for t, h, s, l in cases:
    run_and_save(t, h, s, l, max_depth=args.depth, include_lda=args.include_lda, include_wordvec=args.include_wordvec, 
      par_query=args.par_query, include_openie=args.include_openie, output_path=output_path)
  create_classifier_data(topic=args.topic, output_path=output_path, classifier_path=classifier_path)

if __name__ == '__main__':
  main()