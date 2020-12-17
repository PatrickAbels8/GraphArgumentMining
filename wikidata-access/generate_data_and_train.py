import os
import json
import itertools as it
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from gensim.models import KeyedVectors
from tqdm import tqdm
from utils.printout import create_and_print_knowledge_stats, init_stats_dict, add_metadata_from_split
from models import baseline_statistic_clf as baseline_stat
import classifiers as clf
import argparse
from utils.helper import make_dirs, get_time, save_as_pickle
import git
import time
from helpers.data_generation.helper import *
from helpers.data_generation.training_data_variations import generate_flattened_full_paths_data, generate_BERT_training_data#,#\
#generate_BERT_sent_training_data

def generate_splits(SAMPLES_DIR, PROCESSED_DIR, dev_size=0.2, test_size = 0.1, random_state=0):
    # 1. GENERATE SPLIT
    """
    {
        "nuclear_energy": {
            "train": [0, 5, 9, 12, ], # train sample indices
            "dev": [1, 2, 3, 4, 6, ], # dev sample indices
            "test": [10, 14, ..] # test sample indices
        },
        "death_penalty": {
            "train": [0, 5, 9, 12, ],
            "dev": [1, 2, 3, 4, 6, ],
            "test": [10, 14, ..]
        },
        ...
    }
    """
    print("Generating splits")
    if Path(PROCESSED_DIR + 'indices_split.json').is_file():
        print("Splits already generated -> skipping")
        with open(PROCESSED_DIR + 'indices_split.json', "r") as f:
            return json.load(f)

    # get data for vocab and later preprocessing
    gold_files = [file for file in os.listdir(SAMPLES_DIR) if os.path.isfile(SAMPLES_DIR+file)]
    split_dict = {}
    for data_file in gold_files:
        with open(SAMPLES_DIR+data_file, "r") as f:
            topic_name = data_file.split("result_")[1].split("_nx")[0]

            # load data and get number of samples for the topic
            topic_data = json.load(f)
            topic_sentences = []
            topic_labels = []

            for sample in topic_data["samples"]:
                topic_sentences.append(sample["sentence"][0])
                topic_labels.append(sample["label"])

        # generate stratified splits (indices)
        X = pd.DataFrame(topic_sentences)
        y = pd.Series(topic_labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=dev_size+test_size,
                                                            random_state=random_state, stratify=y)
        X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test,
                                                        test_size=test_size/(dev_size+test_size),
                                                        random_state=random_state, stratify=y_test)
        split_dict[topic_name] = {
            "train_indices": X_train.index.values.tolist(),
            "dev_indices": X_dev.index.values.tolist(),
            "test_indices": X_test.index.values.tolist()
        }

    # save split to disk
    with open(PROCESSED_DIR + 'indices_split.json', 'w') as outfile:
        json.dump(split_dict, outfile, indent=4, sort_keys=True)
        print("Splits file saved as " + PROCESSED_DIR + 'indices_split.json')
        return split_dict

def create_from_preset_splits(DATA_DIR, SAMPLES_DIR, PROCESSED_DIR):
    split_dict = {}
    print("Generating splits from preset split")
    if Path(PROCESSED_DIR + 'preset_indices_split.json').is_file():
        print("Splits from preset split already generated -> skipping")
        with open(PROCESSED_DIR + 'preset_indices_split.json', "r") as f:
            split_dict = json.load(f)
    else:
        # get data for vocab and later preprocessing
        gold_files = [file for file in os.listdir(SAMPLES_DIR) if os.path.isfile(SAMPLES_DIR + file)]
        for data_file in sorted(gold_files):
            topic_name = data_file.split("result_")[1].split("_nx")[0]

            # load knowledge file and check if it has the same length
            with open(SAMPLES_DIR+data_file, "r") as f:
                topic_data = json.load(f)
                train_indices, dev_indices, test_indices = [], [], []
                for i, sample in enumerate(topic_data['samples']):
                    if sample['set'] == "train":
                        train_indices.append(i)
                    elif sample['set'] == "val":
                        dev_indices.append(i)
                    elif sample['set'] == "test":
                        test_indices.append(i)
                    else:
                        print("No set for sentence!")

                split_dict[topic_name] = {
                    "train_indices": train_indices,
                    "dev_indices": dev_indices,
                    "test_indices": test_indices
                }

    # save split to disk
    with open(PROCESSED_DIR + 'preset_indices_split.json', 'w') as outfile:
        json.dump(split_dict, outfile, indent=4, sort_keys=True)
        print("Splits file saved as " + PROCESSED_DIR + 'preset_indices_split.json')

    return split_dict

def generate_training_data(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SHALLOW_PATHS_DIR,
                           PROCESSED_FULL_PATHS_DIR, PROCESSED_SHALLOW_KNOWLEDGE_DIR, random_state=0,
                           combine_topic_and_enity_paths=False, **kwargs):
    # 2. Generate vocab, word_to_index, and index_to_vector
    print("Checking if vocabs were already generated")
    if Path(PROCESSED_DIR + 'vocab_kge.pkl').is_file() and Path(PROCESSED_DIR + 'vocab_we.pkl').is_file()\
            and Path(PROCESSED_DIR + 'wikiLabel_to_wikiId.json').is_file():
        print("Stored vocabs found on the disk: Assuming that training data (except for embeddings lookups) were already created.")
        with open(PROCESSED_DIR + 'vocab_kge.pkl', 'rb') as handle:
            vocab_kge = pickle.load(handle)
        with open(PROCESSED_DIR + 'vocab_we.pkl', 'rb') as handle:
            vocab_we = pickle.load(handle)
        with open(PROCESSED_DIR + 'wikiLabel_to_wikiId.json', 'r') as outfile:
            wikiLabel_to_wikiId_all = json.load(outfile)
        return vocab_we, vocab_kge, wikiLabel_to_wikiId_all

    #############################
    #   1. GENERATE VOCAB       #
    #############################
    print("Vocabs cannot be found on the disk: Loading and processing all topics of corpus")
    # get data for vocab and later preprocessing
    print(os.listdir(SAMPLES_DIR))
    gold_files = [file for file in os.listdir(SAMPLES_DIR) if os.path.isfile(SAMPLES_DIR+file) and file.startswith("result_")]
    training_data = {}
    sentences = []
    wikiLabel_to_wikiId_all = {}
    for data_file in tqdm(gold_files):
        with open(SAMPLES_DIR+data_file, "r") as f:
            topic_name = data_file.split("result_")[1].split("_nx")[0]
            print("Topic: ", topic_name)
            # load data and get number of samples for the topic
            topic_data = json.load(f)
            topic_sentences = []
            topic_labels = []

            # shallow_knowledge
            topic_knowledge_t_dicts = []
            topic_knowledge_s_dicts = []

            # shallow_paths and full_paths
            topic_paths_t_dicts = []
            topic_paths_s_dicts = []
            for sample in topic_data["samples"]:

                # get and add sentence and label to list of all sentences for this topic
                topic_sentences.append(sample["sentence"])
                topic_labels.append(sample["label"])

                # get lookup dict for paths to topic: token -> [predicate, object, predicate, object, ..., token]
                # todo add more tags
                temp_topic_paths_t_dicts, wikiLabel_to_wikiId_all = get_knowledge_lookups(
                    sample["paths_to_topic"], wikiLabel_to_wikiId_all, kwargs['vocab_tag_ent'], kwargs['vocab_tag_rel'],
                    kwargs['vocab_tag_ento'], kwargs['vocab_tag_relo'], kwargs['vocab_tag_open'], kwargs['vocab_tag_wiki'],
                    kwargs['vocab_tag_tok'], all_concepts=False, knowledge_base=kwargs['knowledge_base'])
                topic_paths_t_dicts.append(temp_topic_paths_t_dicts)

                # get lookup dict for paths within sentences: token -> [predicate, object, predicate, object, ..., token]
                temp_topic_paths_s_dicts, wikiLabel_to_wikiId_all = get_knowledge_lookups(
                    sample["paths_between_entities"], wikiLabel_to_wikiId_all, kwargs['vocab_tag_ent'], kwargs['vocab_tag_rel'],
                    kwargs['vocab_tag_ento'], kwargs['vocab_tag_relo'], kwargs['vocab_tag_open'], kwargs['vocab_tag_wiki'],
                    kwargs['vocab_tag_tok'], all_concepts=False, knowledge_base=kwargs['knowledge_base'])
                topic_paths_s_dicts.append(temp_topic_paths_s_dicts)

                # get lookup dict for all concept tokens of the topic: token -> [predicate, object]
                temp_topic_knowledge_t_dicts, wikiLabel_to_wikiId_all = get_knowledge_lookups(
                    sample["topic_concepts"], wikiLabel_to_wikiId_all, kwargs['vocab_tag_ent'], kwargs['vocab_tag_rel'],
                    kwargs['vocab_tag_ento'], kwargs['vocab_tag_relo'], kwargs['vocab_tag_open'], kwargs['vocab_tag_wiki'],
                    kwargs['vocab_tag_tok'], all_concepts=True, knowledge_base=kwargs['knowledge_base'])
                topic_knowledge_t_dicts.append(temp_topic_knowledge_t_dicts)

                # get lookup dict for all concept tokens of the sentence: token -> [predicate, object]
                temp_topic_knowledge_s_dicts, wikiLabel_to_wikiId_all = get_knowledge_lookups(
                    sample["sent_concepts"], wikiLabel_to_wikiId_all, kwargs['vocab_tag_ent'], kwargs['vocab_tag_rel'],
                    kwargs['vocab_tag_ento'], kwargs['vocab_tag_relo'], kwargs['vocab_tag_open'], kwargs['vocab_tag_wiki'],
                    kwargs['vocab_tag_tok'], all_concepts=True, knowledge_base=kwargs['knowledge_base'])
                topic_knowledge_s_dicts.append(temp_topic_knowledge_s_dicts)

                # add to sentence list for vocab creation
                sentences.append(sample["sentence"])

            training_data[topic_name] = {
                "sentence": topic_sentences,
                "label": topic_labels,
                "paths_between_entities": topic_paths_s_dicts,
                "paths_to_topic": topic_paths_t_dicts,
                "sent_concepts": topic_knowledge_s_dicts,
                "topic_concepts": topic_knowledge_t_dicts
            }

    # generate vocab for word embeddings
    print("Generating and saving vocab for word embeddings")
    ngram_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, lowercase=False)  # lowercase=False, decode_error='ignore', encoding='utf',
    ngram_vectorizer.fit_transform(sentences)
    vocab_we = prepare_vocab(ngram_vectorizer.vocabulary_)
    save_as_pickle(vocab_we, PROCESSED_DIR+"vocab_we.pkl")
    print("=> Done!")

    # generate vocab for knowledge graph embeddings
    print("Generating and saving vocab for knowledge graph embeddings")
    vocab_kge = {}
    for i, token in enumerate(wikiLabel_to_wikiId_all.keys()):
        vocab_kge[token] = i+1
    vocab_kge['UNK'] = max(vocab_kge.values())+1
    save_as_pickle(vocab_kge, PROCESSED_DIR+"vocab_kge.pkl")
    print("=> Done!")

    # saving wikiLabel_to_wikiId dictionary
    with open(PROCESSED_DIR + 'wikiLabel_to_wikiId.json', 'w') as outfile:
        json.dump(wikiLabel_to_wikiId_all, outfile, indent=4, sort_keys=True)

    time.sleep(10)
    ############################################
    #     2. sents and knowledge to indices    #
    ############################################
    print("Generating sent_to_indices for each topic")
    # add metadata for max paths/concept lengths
    training_data['max_lengths'] = {
        "shallow_knowledge": defaultdict(int),
        "shallow_paths": defaultdict(int),
        "full_paths": defaultdict(int),
        "sentences": defaultdict(int),

        # store how often specific lengths of concepts/paths appear
        # per_token: not implemented yet
        "distributions": {
            "sentences": {"per_token": defaultdict(int),"per_sent": defaultdict(int)},

            "shallow_paths_topic_sent": {"per_token": defaultdict(int),"per_sent": defaultdict(int)},
            "shallow_paths_sent": {"per_token": defaultdict(int),"per_sent": defaultdict(int)},
            "shallow_paths_topic": {"per_token": defaultdict(int),"per_sent": defaultdict(int)},

            "shallow_knowledge_sent": {"per_token": defaultdict(int),"per_sent": defaultdict(int)},

            "full_paths_topic_sent": {
                "max_paths": {"per_token": defaultdict(int),"per_sent": defaultdict(int)},
                "max_path_len": {"per_token": defaultdict(int),"per_sent": defaultdict(int)},
            }
        }
    }

    temp_index_dict = defaultdict(dict)
    for topic, values in tqdm(training_data.items()):
        if topic == "max_lengths":
            continue

        temp_index_dict[topic] = {
            "all_sent_to_indices": [],

            # shallow_path to indices
            "all_shallow_paths_to_indices": [],
            "all_shallow_paths_to_indices_entities": [],
            "all_shallow_paths_to_indices_topic": [],

            # shallow_knowledge to indices
            "all_shallow_knowledge_to_indices": [],

            # full_paths to indices
            "all_full_paths_to_indices": []
        }

        print("Processing", len(values['sentence']), "sentences for a topic:", topic)
        for i in tqdm(range(len(values['sentence']))):
            # TODO Current simplification: Combine paths to topic and paths between entities (no diff. between them for now)
            """ 
                1. Go through all knowledge paths and get the position where the starting concepts appear in the sample sentence
                2. At this position, write the KGE index of the concept's object (spo-triples) in the knowledge_indices
                Example: 
                    sent = "Is marijuana legalization really awesome against pain streaks ?"
                    sent_indices = [6, 5, 3, 1, 2, 8, 11, 12, 14]
                    vocab_we = {"marijuana": 5, "legalization": 3, "pain": 11, "streaks": 12}
                    vocab_kge = {"Law": 6, "ritalin": 3}
                    knowledge = {"marijuana legalization": [["WIKIFIERED", "Law"], ["WIKIFIERED", "Gesetz"]],
                                 "pain streaks": [["WIKIFIERED", "ritalin"]]}
                => [0, 6, 6, 0, 0, 0, 3, 3, 0]
            """
            ## combine topic and sentence paths: ##
            # get paths knowledge
            paths_between_entities = values['paths_between_entities'][i]
            paths_to_topic = values['paths_to_topic'][i]

            # get combined paths
            combined_paths = combine_topic_and_sent_paths(paths_to_topic, paths_between_entities)

            ###############################
            #     2.0 sents_to_indices    #
            ###############################
            # convert sentences to indices and append to final array
            sent_to_indices = [vocab_we[w] for w in nltk.word_tokenize(values['sentence'][i])]
            update_max_lengths_dict(training_data['max_lengths']['sentences'], sent_to_indices,
                                    training_data['max_lengths']['distributions']['sentences'],
                                    prefix="max_sent_len", setting="")

            temp_index_dict[topic]["all_sent_to_indices"].append(sent_to_indices)

            #######################################
            #     2.1 shallow_paths_to_indices    #
            #######################################
            if combine_topic_and_enity_paths == True: # make no differentiation between paths between entities and paths to topic => combine them
                # get shallow_path to indices vectors for sentences and topics combined and append to all
                shallow_paths_to_indices = get_knowledge_to_indices_for_sent(sent_to_indices, values['sentence'][i], combined_paths,
                                                                             vocab_we, vocab_kge, kwargs['knowledge_base']) # TODO KeyError: 'formal uniform|_e'

                update_max_lengths_dict(training_data['max_lengths']['shallow_paths'], shallow_paths_to_indices,
                                            training_data['max_lengths']['distributions']['shallow_paths_topic_sent'], prefix="max_concept", setting="topic_sent")
                temp_index_dict[topic]["all_shallow_paths_to_indices"].append(shallow_paths_to_indices)

            else:
                # get shallow_path to indices vectors for sentences and topics separately and append to all
                shallow_paths_to_indices_entities = get_knowledge_to_indices_for_sent(sent_to_indices, values['sentence'][i], paths_between_entities,
                                                                                      vocab_we, vocab_kge, kwargs['knowledge_base'])
                shallow_paths_to_indices_topic = get_knowledge_to_indices_for_sent(sent_to_indices, values['sentence'][i], paths_to_topic,
                                                                                   vocab_we, vocab_kge, kwargs['knowledge_base'])
                update_max_lengths_dict(training_data['max_lengths']['shallow_paths'], shallow_paths_to_indices_entities,
                                            training_data['max_lengths']['distributions']['shallow_paths_sent'], prefix="max_concept", setting="sent")
                update_max_lengths_dict(training_data['max_lengths']['shallow_paths'], shallow_paths_to_indices_topic,
                                            training_data['max_lengths']['distributions']['shallow_paths_topic'], prefix="max_concept", setting="topic")
                temp_index_dict[topic]["all_shallow_paths_to_indices_entities"].append(shallow_paths_to_indices_entities)
                temp_index_dict[topic]["all_shallow_paths_to_indices_topic"].append(shallow_paths_to_indices_topic)
                assert training_data['max_lengths']['shallow_paths']['max_concepts_topic_sent'] == \
                       max(training_data['max_lengths']['shallow_paths']['max_concepts_topic'],
                           training_data['max_lengths']['shallow_paths']['max_concepts_sent']), \
                    "Max. path len of the combined path list and the max(topic_path_lens, sentence_path_lens) should be equal"

            ###########################################
            #     2.2 shallow_knowledge_to_indices    #
            ###########################################
            # get shallow_knowledge to indices vectors for sentences append to all (todo for now it doesnt make sense for topics)
            shallow_knowledge_to_indices = get_knowledge_to_indices_for_sent(sent_to_indices, values['sentence'][i],
                                                                             values['sent_concepts'][i], vocab_we, vocab_kge,
                                                                             kwargs['knowledge_base'])
            update_max_lengths_dict(training_data['max_lengths']['shallow_knowledge'], shallow_knowledge_to_indices,
                                        training_data['max_lengths']['distributions']['shallow_knowledge_sent'], prefix="max_concept", setting="sent")
            temp_index_dict[topic]["all_shallow_knowledge_to_indices"].append(shallow_knowledge_to_indices)

            ####################################
            #     2.3 full_paths_to_indices    #
            ####################################
            full_paths_to_indices = get_knowledge_to_indices_for_sent(sent_to_indices, values['sentence'][i], combined_paths,
                                                                      vocab_we, vocab_kge, kwargs['knowledge_base'], full_paths=True)
            
            update_max_lengths_dict(training_data['max_lengths']['full_paths'], full_paths_to_indices,
                                        training_data['max_lengths']['distributions']['full_paths_topic_sent'], prefix="max_path", setting="topic_sent")
            temp_index_dict[topic]["all_full_paths_to_indices"].append(full_paths_to_indices)



    # if max lengths for the number of concepts, paths and path lengths are given, restrict the calculated lengths and update dict
    restrict_max_lengths_dict(training_data['max_lengths'], kwargs['max_concepts'],kwargs['max_paths'],
                              kwargs['max_path_len'], kwargs['max_sent_len'])

    # save the distributions for visualization 
    save_max_len_distributions(training_data['max_lengths']['distributions'], PROCESSED_DIR, granularity="per_sent")

    ##########################
    #     3. Pad sequences   #
    ##########################
    print("Start padding sequences for all topics and saving them to disk")
    for topic, values in tqdm(temp_index_dict.items()):

        ######## shallow_paths ######### pad all shallow_path sequences, create numpy arrays and save them
        print("Padding shallow path matrix for topic: " + str(topic))
        if combine_topic_and_enity_paths == True:
            all_shallow_paths_to_indices = pad_shallow_matrices(values["all_shallow_paths_to_indices"], kwargs['max_sent_len'],
                                                               training_data['max_lengths']['shallow_paths']['max_concepts_topic_sent'])
            np.save(PROCESSED_SHALLOW_PATHS_DIR + topic + "_kX.npy", all_shallow_paths_to_indices)
        else: # if path and topic knowledge is distinguihsed
            all_shallow_paths_to_indices_entities = pad_shallow_matrices(values["all_shallow_paths_to_indices_entities"], kwargs['max_sent_len'],
                                                               training_data['max_lengths']['shallow_paths']['max_concepts_sent'])
            all_shallow_paths_to_indices_topic = pad_shallow_matrices(values["all_shallow_paths_to_indices_topic"], kwargs['max_sent_len'],
                                                               training_data['max_lengths']['shallow_paths']['max_concepts_topic'])
            np.save(PROCESSED_SHALLOW_PATHS_DIR + topic + "_entities_kX.npy", all_shallow_paths_to_indices_entities)
            np.save(PROCESSED_SHALLOW_PATHS_DIR + topic + "_topic_kX.npy", all_shallow_paths_to_indices_topic)
    
        ######## shallow_knowledge ######### pad shallow_knowledge sequences, create numpy arrays and save them
        print("Padding shallow knowledge matrix for topic: " + str(topic))
        all_shallow_knowledge_to_indices = pad_shallow_matrices(values["all_shallow_knowledge_to_indices"], kwargs['max_sent_len'],
                                                            training_data['max_lengths']['shallow_knowledge'][
                                                                'max_concepts_sent'])
        np.save(PROCESSED_SHALLOW_KNOWLEDGE_DIR + topic + "_kX.npy", all_shallow_knowledge_to_indices)

        ######## full_paths #########
        print("Padding full path matrix for topic: " + str(topic))
        all_full_paths_to_indices = pad_full_matrices(values["all_full_paths_to_indices"], kwargs['max_sent_len'],
                                                      training_data['max_lengths']['full_paths']['max_paths_topic_sent'],
                                                      training_data['max_lengths']['full_paths']['max_path_len_topic_sent'])
        np.save(PROCESSED_FULL_PATHS_DIR + topic + "_kX.npy", all_full_paths_to_indices)

        ######## sentence sequences ######### pad sentence sequences, create numpy arrays and save them
        all_sent_to_indices = pad_sequences(values["all_sent_to_indices"], maxlen=training_data['max_lengths']['sentences']['max_sent_len'],
                                            dtype='int32', padding='pre', truncating='pre', value=0.0)
        print("Saving all_sent_to_indices, which are: ", all_sent_to_indices)
        np.save(PROCESSED_DIR+topic+"_X.npy", all_sent_to_indices)

    # return vocabs
    del temp_index_dict
    return vocab_we, vocab_kge, wikiLabel_to_wikiId_all

def generate_index_to_vectors_we(vocab_we, stats_dict, we_setting, PROCESSED_DIR, debug=True):
    #############################
    #      4. index_to_vec      #
    #############################

    print("Generating index_to_vec for WE")
    if Path(PROCESSED_DIR + "index_to_vec_we"+we_setting[1]+".npy").is_file():
        print("Index_to_vec for WE already generated -> skipping")
    else:
        from gensim.scripts.glove2word2vec import glove2word2vec
        # A. For word embeddings
        # load word embeddings and get some statistical info needed
        if "glove" in we_setting[0]:
            glove2word2vec(we_setting[0], we_setting[0][:-4]+"-w2v.txt")
            word_vecs_we = KeyedVectors.load_word2vec_format(we_setting[0][:-4]+"-w2v.txt", binary=False)
        else:
            word_vecs_we = KeyedVectors.load_word2vec_format(we_setting[0], binary=True)
        word_vecs_max = np.max(word_vecs_we.vectors)
        word_vecs_min = np.min(word_vecs_we.vectors)

        # init index_to_vecs with zeros and randomize "UNK" vector at last pos
        index_to_vec_we = np.zeros((len(vocab_we.items())+1, word_vecs_we.vector_size))
        index_to_vec_we[vocab_we['UNK']] = np.random.uniform(word_vecs_min, word_vecs_max, word_vecs_we.vector_size)
        oov_counter = 0
        for token, index in vocab_we.items():# Unknown words of unseen data (not in vocab) will all get the same vector!
            vec, vec_found = get_we_vector_for_token(token, word_vecs_we, word_vecs_min, word_vecs_max, debug=debug)
            if vec_found == False:
                vec, vec_found = get_we_vector_for_token(token.lower(), word_vecs_we, word_vecs_min, word_vecs_max, debug=debug)
            if vec_found == False and len(token) > 1:
                vec, vec_found = get_we_vector_for_token(token[0].upper() + token[1:].lower(), word_vecs_we, word_vecs_min, word_vecs_max, debug=debug)
                if vec_found == False:
                    oov_counter += 1
            index_to_vec_we[index] = vec

        stats_dict['metadata']['oov_we_abs'] = oov_counter
        stats_dict['metadata']['we_length'] = len(vocab_we.items())
        stats_dict['metadata']['oov_we_rel'] = "{0:.2f}".format((oov_counter / len(vocab_we.items())) * 100)
        if debug == True:
            print(str(oov_counter)+"/"+str(len(vocab_we.items()))+" words are not in the word embeddings")

        # save to disk
        np.save(PROCESSED_DIR + "index_to_vec_we"+we_setting[1]+".npy", index_to_vec_we)

def generate_index_to_vectors_kge( vocab_kge, wikiLabel_to_wikiId, stats_dict, kg_setting, PROCESSED_DIR, debug=True, **kwargs):

    #############################
    #      4. index_to_vec      #
    #############################
    print("Generating index_to_vec for KGE")
    if Path(PROCESSED_DIR + "index_to_vec_kge"+kg_setting[1]+".npy").is_file():
        print("Index_to_vec for KGE already generated -> skipping")
    else:
        # check for w2v files
        is_w2v = False
        if "conceptnet" in kg_setting[0] or "babelnet" in kg_setting[0]:
            if kg_setting[0].endswith(".bin"):
                entity_w2v = KeyedVectors.load_word2vec_format(kg_setting[0], binary=True)
            else:
                entity_w2v = KeyedVectors.load_word2vec_format(kg_setting[0], binary=False)
            relation_w2v = None
            word_vecs_max = np.max(entity_w2v.vectors)
            word_vecs_min = np.min(entity_w2v.vectors)
            is_w2v = True
        else:
            if os.path.isfile(kg_setting[0] + "entity_w2v.bin") and \
                    os.path.isfile(kg_setting[0] + "relation_w2v.bin"):
                entity_w2v = KeyedVectors.load_word2vec_format(kg_setting[0] + "entity_w2v.bin", binary=True)
                relation_w2v = KeyedVectors.load_word2vec_format(kg_setting[0] + "relation_w2v.bin", binary=True)
                word_vecs_max = np.max([np.max(entity_w2v.vectors), np.max(relation_w2v.vectors)])
                word_vecs_min = np.min([np.min(entity_w2v.vectors), np.min(relation_w2v.vectors)])
                is_w2v = True
            else:
                # load entity2id and id2vec for entities
                entityId_to_vecindex = load_wikiId_to_vec(kg_setting[0] + "entity2id.txt")
                vecindex_to_entvec = pd.read_csv(kg_setting[0]+"entity2vec.vec", delimiter="\t",
                                              header=None, usecols=range(kg_setting[2])).values

                # load entity2id and id2vec and merge tables for relations
                relationId_to_vecindex = load_wikiId_to_vec(kg_setting[0] + "relation2id.txt")
                vecindex_to_relvec = pd.read_csv(kg_setting[0]+"relation2vec.vec", delimiter="\t",
                                              header=None, usecols=range(kg_setting[2])).values

                # min/max  for random vector
                word_vecs_max = np.max([np.max(vecindex_to_entvec), np.max(vecindex_to_relvec)])
                word_vecs_min = np.min([np.min(vecindex_to_entvec), np.min(vecindex_to_relvec)])

        # init index_to_vecs with zeros and randomize "UNK" vector at last pos
        index_to_vec_kge = np.zeros((len(vocab_kge.items()) + 1, kg_setting[2]))
        index_to_vec_kge[vocab_kge['UNK']] = np.random.uniform(word_vecs_min, word_vecs_max, kg_setting[2])
        oov_counter = 0
        for wikiLabel, index in vocab_kge.items():  # TODO Unknown words at testing time will all get the same vector => use whole
            if wikiLabel == 'UNK': # "UNK"
                continue

            wikiId = wikiLabel_to_wikiId.get(wikiLabel, None) # get wikiId (e.g. Q76) for label (e.g. "Obama|_e")

            if is_w2v == False:
                index_to_vec_kge[index], vec_found = get_kge_vector_for_entity(entityId_to_vecindex, relationId_to_vecindex,
                                                                               vecindex_to_entvec, vecindex_to_relvec, wikiLabel,
                                                                               wikiId, kwargs['vocab_tag_ent'], kwargs['vocab_tag_open'], kwargs['vocab_tag_wiki'],
                                                                               kwargs['vocab_tag_rel'], kwargs['vocab_tag_ento'], kwargs['vocab_tag_relo'],
                                                                               kwargs['vocab_tag_tok'], word_vecs_min, word_vecs_max,
                                                                               kg_setting[2])
            else:
                index_to_vec_kge[index], vec_found = get_kge_vector_for_entity_w2v(wikiLabel, wikiId, entity_w2v, relation_w2v,
                                                                                   kwargs['vocab_tag_ent'], kwargs['vocab_tag_open'], kwargs['vocab_tag_wiki'],
                                                                                   kwargs['vocab_tag_rel'], kwargs['vocab_tag_ento'], kwargs['vocab_tag_relo'],
                                                                                   kwargs['vocab_tag_tok'], word_vecs_min, word_vecs_max, debug=debug)
            if vec_found == False:
                oov_counter += 1


        stats_dict['metadata']['oov_kge_abs'] = oov_counter
        stats_dict['metadata']['kge_length'] = len(vocab_kge.items())
        stats_dict['metadata']['oov_kge_rel'] = "{0:.2f}".format((oov_counter / len(vocab_kge.items()))*100)
        if debug == True:
            print(str(oov_counter) + "/" + str(len(vocab_kge.items())) + " entities are not in the knowledge graph embeddings")

        # save to disk
        np.save(PROCESSED_DIR + "index_to_vec_kge"+kg_setting[1]+".npy", index_to_vec_kge)

    # TODO bug in sentence saved: has also the label appended
    #  create method that returns text get_text(topic, index, json_file)

def start_training_process(model_settings, SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, RESULTS_DIR, PREDICT_TEST):
    """
    Starts the training process and handles tuning, if lists of parameters are given
    :param model_settings:
    :param SAMPLES_DIR:
    :param PROCESSED_DIR:
    :param RESULTS_DIR:
    :param PREDICT_TEST:
    :return:
    """
    # possible tuning params: word_embeddings, dropout, lstm_size, batch_size, learning_rate
    tuning_dict = {}

    if all(isinstance(el, list) for el in model_settings['word_embeddings']):
        tuning_dict["1_word_embeddings"] = model_settings['word_embeddings']
    if isinstance(model_settings['dropout'], list):
        tuning_dict["2_dropout"] = model_settings['dropout']
    if isinstance(model_settings['lstm_size'], list):
        tuning_dict["3_lstm_size"] = model_settings['lstm_size']
    if isinstance(model_settings['batch_size'], list):
        tuning_dict["4_batch_size"] = model_settings['batch_size']
    if isinstance(model_settings['learning_rate'], list):
        tuning_dict["5_learning_rate"] = model_settings['learning_rate']

    if len(tuning_dict.keys()) == 0: # no hyperparameter tuning detected => train mdoel as usual
        clf.train_clf(model_settings['model'], SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, RESULTS_DIR,
                      model_settings=model_settings, predict_test=PREDICT_TEST)
    else:
        # get all combinations for params
        sorted_param_names = sorted(tuning_dict)
        param_combinations = it.product(*(tuning_dict[param_name] for param_name in sorted_param_names))

        # list that holds the model_settings with all parameter configs to go through
        model_settings_list = []
        for param_comb in param_combinations:
            updated_values = {p[2:]: c for p, c in list(zip(sorted_param_names, param_comb))}
            new_model_setting = copy.deepcopy(model_settings)
            new_model_setting.update(updated_values)
            model_settings_list.append(new_model_setting)

        # run all model_settings
        for setting in model_settings_list:
            clf.train_clf(setting['model'], SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, RESULTS_DIR,
                          model_settings=setting, predict_test=PREDICT_TEST)

if __name__ == '__main__':
    # Results: https://docs.google.com/spreadsheets/d/126rMZtVsvkL0SukOaD8y3FbgJ6MPGw_qZhRSp5uIpK0/edit#gid=0

    # 2677/29202 words are not in the word embeddings (glove)
    # oSaI:
        # 6085/17514 wiki entities are not in the knowledge graph embeddings (daniil's)
        # ~2000/17514 wiki entities are not in the knowledge graph embeddings (benjamin's)
    # oSaI limit 15:
        # 5056/28300 wiki entities are not in the knowledge graph embeddings (benjamin's)

    # load values from config if given
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--config', type=str, help='path to config.json')
    parser.add_argument('--predict-test', type=str, help='enter 1 if test results should be stored, too')
    parser.add_argument('--bert-server', type=str, help='BERT embedding server ip')
    args = parser.parse_args()
    PREDICT_TEST = True if args.predict_test == "1" else False
    BERT_SERVER_IP = "localhost" if args.bert_server == None else args.bert_server

    try:  # todo get repo infos, remove for "production"
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        branch = repo.head.ref.name
    except:
        sha = None
        branch = None

    # set default values
    model_settings = {
        "dropout": 0.3,
        "lstm_size": 32,
        "monitor": "val_loss",
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 0.001,
        "train_embeddings": False,
        "label_setup": "two_label",
        "train_setup": "cross_domain",
        "model": "EvLSTM",
        "knowledge_enriched_data": "only_sub_and_inst",
        "combine_topic_and_entity_paths": True,
        "knowledge_config": "shallow_paths",
        "word_embeddings": ["embeddings/en/google_news/GoogleNews-vectors-negative300.bin", "_gnews300", 300],
        "kg_embeddings": ["embeddings/en/kge_daniil/dec_17_100/", "_wiki_daniil_100", 100],
        "knowledge_base": "wikidata_cypher",
        "max_sent_len": 60,
        "max_concepts": 4,
        "max_paths": 5,
        "max_path_len": 11,  # should be odd numbers, if relations are included ([subj] = 1,  [subj,pred,obj] = 3, ...)
        "git_branch": branch,
        "git_sha": sha,
        "training_start": get_time(),
        "dataset": "UKPSententialArgMin"

    }

    # load config passed via CLI
    if args.config != None and args.config != "":
        with open(args.config) as f:
            model_settings.update(json.load(f))

    # CONSTANTS
    SAMPLES_DIR = "results/" + model_settings["dataset"] + "/" + model_settings["knowledge_base"] + "/" \
                  + model_settings['knowledge_enriched_data'] + "/"
    DATA_DIR = "data/en/" + model_settings["dataset"] + "/tsv/"
    PROCESSED_DIR = SAMPLES_DIR + "processed/"
    PROCESSED_SHALLOW_PATHS_DIR = PROCESSED_DIR + "shallow_paths/"
    PROCESSED_FULL_PATHS_DIR = PROCESSED_DIR + "full_paths/"
    PROCESSED_FULL_PATHS_FLATTENED_DIR = PROCESSED_DIR + "full_paths_flattened/"
    BERT_WORD_EMB_SENTS_DIR = PROCESSED_DIR + "bert_word_emb/"
    BERT_SENT_EMB_SENTS_DIR = PROCESSED_DIR + "bert_sent_emb/"
    PROCESSED_SHALLOW_KNOWLEDGE_DIR = PROCESSED_DIR + "shallow_knowledge/"
    RESULTS_DIR = SAMPLES_DIR + "model_runs/"
    DEBUG = True
    USE_PRESET_SPLITS = True
    CREATE_STATS = False
    PRINTOUT_FORMAT = "tex"
    STAT_BASELINE = False
    VOCAB_TAG_ENT = "|_e"
    VOCAB_TAG_REL = "|_r"
    VOCAB_TAG_OPEN = "|_O"
    VOCAB_TAG_WIKI = "|_W"
    VOCAB_TAG_ENTo = "|_E"
    VOCAB_TAG_RELo = "|_R"
    VOCAB_TAG_TOK = "|_T"

    # Create needed folder(s)
    make_dirs([PROCESSED_SHALLOW_PATHS_DIR, PROCESSED_FULL_PATHS_DIR, PROCESSED_SHALLOW_KNOWLEDGE_DIR])

    # create stats dict (add new statistic values here)
    stats_dict = init_stats_dict(SAMPLES_DIR, format=PRINTOUT_FORMAT, dataset=model_settings['dataset'])

    # generate splits
    if USE_PRESET_SPLITS == True:
        # TODO should be done with hashs then instead of indices
        split_dict = create_from_preset_splits(DATA_DIR, SAMPLES_DIR, PROCESSED_DIR)
    else:
        split_dict = generate_splits(SAMPLES_DIR, PROCESSED_DIR, dev_size=0.2, test_size=0.1, random_state=0)

    # add information about split sizes to metadata
    add_metadata_from_split(split_dict, stats_dict)

    # generate vocabs and sent_to_indices
    # todo send more tags
    
    vocab_we, vocab_kge, wikiLabel_to_wikiId = generate_training_data(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SHALLOW_PATHS_DIR,
                                                                      PROCESSED_FULL_PATHS_DIR, PROCESSED_SHALLOW_KNOWLEDGE_DIR,
                                                                      random_state=0, combine_topic_and_enity_paths=model_settings['combine_topic_and_entity_paths'],
                                                                      max_concepts=model_settings['max_concepts'], max_paths=model_settings['max_paths'],
                                                                      max_path_len=model_settings['max_path_len'], max_sent_len=model_settings['max_sent_len'],
                                                                      vocab_tag_ent=VOCAB_TAG_ENT, vocab_tag_rel=VOCAB_TAG_REL,
                                                                      vocab_tag_open=VOCAB_TAG_OPEN, vocab_tag_wiki=VOCAB_TAG_WIKI, vocab_tag_relo=VOCAB_TAG_RELo, 
                                                                      vocab_tag_ento=VOCAB_TAG_ENTo, vocab_tag_tok=VOCAB_TAG_TOK,
                                                                      knowledge_base=model_settings['knowledge_base'])

    # generate index_to_vectors for kge
    if all(isinstance(el, list) for el in model_settings['kg_embeddings']):
        for kg_setting in model_settings['kg_embeddings']:
            generate_index_to_vectors_kge(vocab_kge, wikiLabel_to_wikiId, stats_dict, kg_setting, PROCESSED_DIR,
                                  debug=DEBUG, vocab_tag_ent=VOCAB_TAG_ENT, vocab_tag_rel=VOCAB_TAG_REL,
                                  vocab_tag_open=VOCAB_TAG_OPEN, vocab_tag_wiki=VOCAB_TAG_WIKI, vocab_tag_relo=VOCAB_TAG_RELo, 
                                  vocab_tag_ento=VOCAB_TAG_ENTo, vocab_tag_tok=VOCAB_TAG_TOK)
    elif "BERT" not in model_settings['kg_embeddings'][0]:
        generate_index_to_vectors_kge(vocab_kge, wikiLabel_to_wikiId, stats_dict, model_settings['kg_embeddings'], PROCESSED_DIR,
                                      debug=DEBUG, vocab_tag_ent=VOCAB_TAG_ENT, vocab_tag_rel=VOCAB_TAG_REL,
                                      vocab_tag_open=VOCAB_TAG_OPEN, vocab_tag_wiki=VOCAB_TAG_WIKI, vocab_tag_relo=VOCAB_TAG_RELo, 
                                      vocab_tag_ento=VOCAB_TAG_ENTo, vocab_tag_tok=VOCAB_TAG_TOK)

    # convert training data sentences to BERT embedded sentences
    if "BERT" in model_settings['kg_embeddings'][0]:
        SOURCE_PATH = ""
        if model_settings['knowledge_config'] == "shallow_knowledge":
            SOURCE_PATH = PROCESSED_SHALLOW_KNOWLEDGE_DIR
        if model_settings['knowledge_config'] == "shallow_paths":
            SOURCE_PATH = PROCESSED_SHALLOW_PATHS_DIR
        if model_settings['knowledge_config'] == "full_paths":
            raise Exception("Full paths as BERT knowlede not supported yet")

        if model_settings['kg_embeddings'][0] == "BERT_KNOWLEDGE_SENT":
            TARGET_DIR = SOURCE_PATH[:-1] + "_bert_sent_emb/"
        else:
            TARGET_DIR = SOURCE_PATH[:-1] + "_bert_word_emb/"

        generate_BERT_training_data(vocab_kge, model_settings['kg_embeddings'], SOURCE_PATH,
                                    TARGET_DIR, BERT_SERVER_IP)

    # generate index_to_vectors for we
    if all(isinstance(el, list) for el in model_settings['word_embeddings']):
        # we might get a list of word_embeddings for hyperparameter tuning
        for we_setting in model_settings['word_embeddings']:
            if we_setting[0] != "BERT":
                generate_index_to_vectors_we(vocab_we, stats_dict, we_setting, PROCESSED_DIR, debug=DEBUG)
    elif "BERT" not in model_settings['word_embeddings'][0]:
        generate_index_to_vectors_we(vocab_we, stats_dict, model_settings['word_embeddings'], PROCESSED_DIR, debug=DEBUG)

    # convert training data sentences to BERT embedded sentences
    BERT_PROCESSING_TARGET_DIR = ""
    if "BERT" in model_settings['word_embeddings'][0]:
        BERT_PROCESSING_TARGET_DIR = BERT_WORD_EMB_SENTS_DIR if model_settings['word_embeddings'][0] == "BERT" \
            else BERT_SENT_EMB_SENTS_DIR
        generate_BERT_training_data(vocab_we, model_settings['word_embeddings'], PROCESSED_DIR,
                                    BERT_PROCESSING_TARGET_DIR, BERT_SERVER_IP)

    # generate flattened full paths training data to use shallow models on full path data
    generate_flattened_full_paths_data(PROCESSED_DIR, PROCESSED_FULL_PATHS_DIR, PROCESSED_FULL_PATHS_FLATTENED_DIR)

    # create statistics
    create_and_print_knowledge_stats(SAMPLES_DIR, PROCESSED_DIR, stats_dict, format=PRINTOUT_FORMAT)

    # use features based on the statistics and use them with a simple classifier (LR) + print f1 macro for in-topic 2-lbl results
    if STAT_BASELINE == True:
        #feature_list = ['bow', 'sent_len', 'tokens_linked_within_sent', 'tokens_linked_to_topic',
        #                'sents_with_paths_between_entities', 'sents_with_path_to_topic',
        #                'include_only_topic', 'include_topic']
        feature_list = ['bow', 'include_only_topic', 'include_topic']
        baseline_stat.train_model(SAMPLES_DIR, model_settings["label_setup"], feature_list)

    # start training and also handle parameter tuning
    start_training_process(model_settings, SAMPLES_DIR, PROCESSED_DIR, BERT_PROCESSING_TARGET_DIR, RESULTS_DIR, PREDICT_TEST)
