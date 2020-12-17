import nltk
import regex as re
import copy
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import numpy as np
import csv
from multiprocessing import Lock, Pool
from bert_serving.client import BertClient
from tqdm import tqdm
import time
from utils.helper import invert_dictionary

print_lock = Lock()



def call_BERT_server_async(X, start_index, end_index, vocab_we, setting, BERT_SERVER_IP):
    def handle_word_embeddings(encoded):
        # remove [CLS] and [SEP] part
        encoded = np.delete(encoded, 0, axis=0)
        encoded = np.delete(encoded, len(sent_tokenized), axis=0)

        # return to pre-padding zeros (BERT does post-padding)
        final = encoded[:len(sent_tokenized)]
        if len(X[i]) - final.shape[0] > 0:
            last = np.zeros((len(X[i]) - final.shape[0], final.shape[1]))
            final = np.vstack((last, final))
        return final

    X_new = []
    try:
        bc = BertClient(ip=BERT_SERVER_IP)  # ip address of the GPU machine
        for i in tqdm(range(start_index, end_index)):
            # return back to token strings
            if setting[0] == "BERT" or setting[0] == "BERT_SENT": # sentences
                sent_tokenized = [vocab_we[s] for s in X[i] if s != 0]

                # encode with BERT embeddings
                encoded = bc.encode([sent_tokenized], is_tokenized=True)[0]

                if setting[0] == "BERT":
                    final = handle_word_embeddings(encoded)  # padding & delete CLS/SEP
                elif setting[0] == "BERT_SENT":
                    final = encoded
            else: # knowledge
                final = []
                for concept_list in X[i]:
                    if max(concept_list) == 0:
                        final.append(np.zeros((4, setting[2]))) #todo get the 4 from params
                    else:
                        encoded = bc.encode([vocab_we[s][:-3] for s in concept_list if s != 0])[0]

                        if len(encoded) == setting[2]:
                            encoded = np.expand_dims(encoded, 0)
                        missing_concepts = 4 - len(encoded)
                        encoded = np.vstack((np.zeros((missing_concepts, setting[2])), encoded))

                        final.append(encoded)



            # add processed embedding sequence for sentence
            X_new.append(final)
        return np.array(X_new)
    except Exception as e:
        with print_lock:
            print(str(e))
        return np.array(X_new)

def convert_X_to_BERT_thread(X, vocab_we, setting, BERT_SERVER_IP):
    pool = Pool(processes=2)
    X_new_1 = pool.apply_async(call_BERT_server_async, args=(X, 0, int(len(X)/2.0), vocab_we, setting, BERT_SERVER_IP, ))
    X_new_2 = pool.apply_async(call_BERT_server_async, args=(X, int(len(X)/2.0), len(X), vocab_we, setting, BERT_SERVER_IP, ))
    pool.close()
    pool.join()

    return np.vstack((X_new_1.get(), X_new_2.get()))

def convert_X_to_BERT(X, inverted_vocab_we, setting, BERT_SERVER_IP):

    # call server and get bert token embeddings => async, i.e. X will be split in half
    if len(X) == 1:
        X_new = call_BERT_server_async(X, 0, 1, inverted_vocab_we, setting, BERT_SERVER_IP)
    elif len(X) == 2:
        X_new = convert_X_to_BERT_thread(X, inverted_vocab_we, setting, BERT_SERVER_IP)
    elif len(X) == 3:
        X_new = call_BERT_server_async(X, 0, 1, inverted_vocab_we, setting, BERT_SERVER_IP)
        X_new_2 = call_BERT_server_async(X, 1, 3, inverted_vocab_we, setting, BERT_SERVER_IP)
        X_new = np.vstack((X_new, X_new_2))
    else:
        X_new = convert_X_to_BERT_thread(X[:int(len(X)/2.0)], inverted_vocab_we, setting, BERT_SERVER_IP)
        X_new_2 = convert_X_to_BERT_thread(X[int(len(X)/2.0):], inverted_vocab_we, setting, BERT_SERVER_IP)
        X_new = np.vstack((X_new, X_new_2))

    assert X.shape[0] == X_new.shape[0], "Difference in training sample size after BERT conversion!"

    return X_new

def find(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.

    find([1, 1, 2], [1, 2])
    1
    => https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm
    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            return i - n + 1
    return -1

def find_sublist_positions(haystack, needle):
    # find all positions of a sublist in a list
    if len(needle) == 0:
        return []

    positions = []
    temp_haystack = haystack.copy()

    while True:
        pos = find(temp_haystack, needle)
        if pos == -1:
            break
        else:
            if len(positions) > 0:
                positions.append(pos + positions[-1] + len(needle))
            else:
                positions.append(pos)
            temp_haystack = temp_haystack[pos + len(needle):]
    return positions

def contains_list(items, target):
    # returns True, if target list is contained in items, else returns False
    if len(items) == 0:
        return False
    else:
        for item in items:
            if (len(item) == len(target)) and (len([i for i, j in zip(item, target) if i == j]) == len(item)):
                return True
    return False

def find_original_token_in_sent(token, sent):
    pos = sent.lower().find(token)
    if pos >= 0:
        return sent[pos:pos+len(token)]
    else:
        return token

def get_knowledge_to_indices_for_sent(sent_to_indices, original_sent, paths, vocab_we, vocab_kge, knowledge_base, full_paths=False):
    """
    Creates the training data for the knowledge. The dimension is [#sent, sent_len, #concepts_per_token]. Will
    be padded to maximum number of concepts over all tokens in the set.
    :param sent_to_indices:
    :param paths:
    :param vocab_we:
    :param vocab_kge:
    :return:
    """

    # create initial knowledge indices
    knowledge_to_indices = [[] for _ in range(len(sent_to_indices))]

    # returns knowledge_to_indices for a given sentence
    for start, value in paths.items():
        #temp_kge_indices = [vocab_we[index] for index in nltk.word_tokenize(start)]
        temp_kge_indices = []
        start_const = start # keep this one for vocab_kge lookup
        if start[-1] == ".": # small hack: nltk's word tokenizer will split e.g. "the U.K." into ["the", "U.K", "."] => "U.K" is not in the vocab
            start += " |"
        for token in nltk.word_tokenize(start):
            if token != "." and token != "|":
                try:
                    index = vocab_we.get(token, None) # at the sentence position of this WE-index, a KGE will be placed
                    if index == None: # due to different ways to retrieve knowledge for the KBs, lowercasing might have applied earlier
                        index = vocab_we.get(token[0].upper() + token[1:].lower(), None)
                    if index == None:
                        index = vocab_we.get(token.lower(), None)
                    if index == None:
                        token = find_original_token_in_sent(token, original_sent) # try to find token as it was back in the sentence
                        index = vocab_we.get(token, None)
                    if index == None:
                        raise(KeyError)
                    temp_kge_indices.append(index)
                except KeyError as ke:
                    print("Token: ", token, "Error:", str(ke) + "; cannot find this key in the WE_vocab, since the support_word found by wikifier is not split in the same"
                                " way as nltk does.")
                    pass
        positions = find_sublist_positions(sent_to_indices, temp_kge_indices) # sublist can appear multiple times

        for pos in positions: # all positions where mention appears in the sentence
            for j in range(len(temp_kge_indices)): # if mention has more than one token, add concept/path for all
                for path_number in range(len(paths[start_const])): # mention can have several paths starting from it
                    if full_paths == False:
                        try:
                            get_first_concept = paths[start_const][path_number][1]
                            get_first_concept_kge_index = vocab_kge[get_first_concept] # TODO KeyError: 'formal uniform|_e'
                            if get_first_concept_kge_index not in knowledge_to_indices[pos + j]:# TODO cannot happen anymore ?! (duplicates)
                                knowledge_to_indices[pos + j].append(get_first_concept_kge_index)
                        except IndexError:
                            pass
                    else:
                        path_to_kge_indices = [vocab_kge[c] for c in paths[start_const][path_number][1:-2]] #todo change here, if we want mentions in the path
                        if not contains_list(knowledge_to_indices[pos + j], path_to_kge_indices):
                            knowledge_to_indices[pos + j].append(path_to_kge_indices)
                        #else:
                        #    print("duplicate in list")
                        #    contains_list(knowledge_to_indices[pos + j], path_to_kge_indices)
    return knowledge_to_indices

def prepare_vocab(vocab):
    # add a 0 for unknown words TODO lower case them??
    for key, value in vocab.items():
        vocab[key] = value + 1
    vocab['UNK'] = max(vocab.values())+1
    return vocab

def remove_ids_from_path(path, vocab_tag_ent, vocab_tag_rel, all_concepts):
    # TODO CON what to do?
    # removes the id from the knowledge base from the paths and adds the entitity/relation tags
    temp_list = []
    for number, entity in enumerate(path):
        if number % 2 == 0:
            tag = vocab_tag_ent if number > 0 and ((number < len(path)-1 and not all_concepts) or (number < len(path) and all_concepts)) else ""
            temp_list.append(entity.rsplit(";", 1)[0] + tag)
        else:
            temp_list.append(entity.rsplit(";", 1)[0] + vocab_tag_rel)
    return temp_list

def get_knowledge_lookups(paths, wikiLabel_to_wikiId_all, vocab_tag_ent, vocab_tag_rel, vocab_tag_ento, vocab_tag_relo, vocab_tag_open, vocab_tag_wiki, vocab_tag_tok, all_concepts=False, knowledge_base="wikidata_cypher"):
    """
    - Converts the path-strings to a list in a dict that contains the starting word of a path and the following path(s)
      Example:  {
                "bad": [
                        ["WIKIFIERED", "Evil", "DIFFERENT_FROM", "good"], ...
                       ]
                }
    - Also returns a list of all words found in the paths
    """
    wikiLabel_to_wikiId = {}
    result_dict = defaultdict(list)
    for path in paths:
        #print("Path before regex: ", path)
        if "wordnet-rdf" in path:
            path = path.split("->")
            path = [p.strip("[").strip("]").strip("(").strip(")") for p in path]
        else:
            if all_concepts == False:
                path = re.findall("[\[|\(]([^>]*;[^>]*)[\]|\)]", path) # https://regexr.com/ (old: [\[|\(]([^>]*;\w*)[\]|\)]   )
            else:
                path = re.findall("[\[|\(]([^>]*[;]*\w*)[\]|\)]", path) # https://regexr.com/      
        #print("Path after regex: ", path)
        # if "wikidata" in knowledge_base or "knowledge_graph" in knowledge_base and wikidata: 
        #     for i, p in enumerate(path): # TODO sometimes there are multiple relations between nodes in one direction => take only the latter
        #         temp_predicates = p.rsplit("/", 1)
        #         if len(temp_predicates) == 2 and \
        #             len(temp_predicates[0].rsplit(";", 1)) == 2 and \
        #             len(temp_predicates[1].rsplit(";", 1)) == 2:
        #             path[i] = p.rsplit("/", 1)[1]
        #             print("Path now: ", path)
        #         #if "p279" in p.lower() and "p31" in p.lower(): # exception: e.g. 9/11
        #         #    path[i] = p.rsplit("/", 1)[1]
                
        
        if all_concepts == False:
            # we don't want the WIKIFIERED relations, nor start/end tokens of the sentence
            '''
            TODO CON

            not only remove first and last two tokens but add more tags:
            - |_e, |_r, |_E, |_R, |_O, |_W, |_T
            vocab_tag_ent
            vocab_tag_rel
            vocab_tag_ento
            vocab_tag_relo
            vocab_tag_open
            vocab_tag_wiki
            vocab_tag_tok
            '''

            '''
            NEW
            '''
            # def getClosestEnt(tok):
            #     return 'Q123' # doesnt matter since 'getkgevector' does the matching stuff
                                # maybe does matter bc line 207 throws error otherwise, so maybe shold be mapped here

            # def getClosestRel(tok):
            #     return 'P123' # doesnt matter since 'getkgevector' does the matching stuff

            # def getDicc(path):
            #     dicc = {}
            #     for number, entity in enumerate(path[2:-2]):
            #         olabel, oid = entity.rsplit(";", 1)

            #         if oid.isnumeric():
            #             nlabel = olabel+vocab_tag_tok
            #         elif 'WIKIFIERED' in olabel:
            #             nlabel = olabel+vocab_tag_wiki
            #         elif '=openie=' in olabel:
            #             nlabel = olabel+vocab_tag_open
            #         elif oid[0]=='Q' and oid[1].isnumeric():
            #             nlabel = olabel+vocab_tag_ent
            #             nid = oid
            #         elif oid[0]=='P' and oid[1].isnumeric():
            #             nlabel = olabel+vocab_tag_rel
            #             nid = oid
            #         elif olabel==oid:
            #             nlabel = olabel+vocab_tag_ento
            #             nid = getClosestEnt(olabel)
            #         else:
            #             nlabel = olabel+vocab_tag_relo
            #             nid = getClosestRel(olabel)
            
            #         try:
            #             dicc[nlabel] = nid
            #         except:
            #             dicc[nlabel] = 'UNK'
            #     return dicc

            # wikiLabel_to_wikiId.update(getDicc(path))
                                        


            '''
            OLD
            '''
            wikiLabel_to_wikiId.update({(entity.rsplit(";", 1)[0]+vocab_tag_ent if number % 2 == 0 else entity.rsplit(";", 1)[0]+vocab_tag_rel)
                                        : entity.rsplit(";", 1)[1]
                                        for number, entity in enumerate(path[2:-2])})  # we don't want the WIKIFIERED relations
        else:
            try:
                lbl, id = path[2].rsplit(";", 1)[0].rsplit(";", 1)
                wikiLabel_to_wikiId[lbl+vocab_tag_ent] = id
            except IndexError:
                print("Index Error occured. Path is: ", path)
                pass

        if all_concepts == True: # pageRank is included to each concepts and is removed here
            try:
                path[2] = path[2].rsplit(";", 1)[0]
            except IndexError:
                print("Index Error occured. Path: ", path)
                pass
        path = remove_ids_from_path(path, vocab_tag_ent, vocab_tag_rel, all_concepts) # remove ids and QIDs and add rel/ent tags for lookup
        
        # add path/concept to list, if no duplicate (use deepcopy, or call will add the key to defaultdict)
        if contains_list(copy.deepcopy(result_dict)[path[0]], path[1:]) == False:
            result_dict[path[0]].append(path[1:])

        # reverse paths and add to list if no duplicate (use deepcopy, or call will add the key to defaultdict)
        if all_concepts == False:
            path.reverse()
            if contains_list(copy.deepcopy(result_dict)[path[0]], path[1:]) == False: # no duplicates
                result_dict[path[0]].append(path[1:])

    wikiLabel_to_wikiId_all.update(wikiLabel_to_wikiId)

    return result_dict, wikiLabel_to_wikiId_all

def combine_topic_and_sent_paths(topic_paths, sent_paths):
    # combines and returns the paths to the topic and in between the sentence
    combined_paths = copy.deepcopy(sent_paths)
    for key, paths in topic_paths.items():
        if key in combined_paths.keys():
            for path in paths:
                if not contains_list(combined_paths[key], path):
                    combined_paths[key].append(path)
        else:
            combined_paths[key] = paths
    return combined_paths

def pad_shallow_matrices(all_shallow_to_indices, MAX_SENT_LEN, MAX_CONCEPTS_LEN):
    for kti_vec_index, kti_vector in enumerate(all_shallow_to_indices):

        # check for errors
        #if max([len(concept_list) for concept_list in kti_vector]) > MAX_CONCEPTS_LEN:
        #    print("Number of concepts for a token was longer than pre-calculated and would be cut. ")

        # do actual padding
        padding_len = MAX_SENT_LEN - len(kti_vector)
        if padding_len > 0:
            kti_vector = [[] for _ in range(padding_len)] + kti_vector
        if padding_len < 0:
            kti_vector = kti_vector[-MAX_SENT_LEN:]
        all_shallow_to_indices[kti_vec_index] = pad_sequences(kti_vector,
                                                              maxlen=MAX_CONCEPTS_LEN, dtype='int32',
                                                              padding='pre', truncating='pre', value=0.0)
    return all_shallow_to_indices

def pad_full_matrices(all_full_to_indices, MAX_SENT_LEN, MAX_PATHS, MAX_PATH_LEN):
    #padded_matrix = np.zeros((len(all_full_to_indices), MAX_SENT_LEN, MAX_PATHS, MAX_PATH_LEN))
    for kti_vec_index, kti_vector in enumerate(all_full_to_indices):

        # check for errors
        #if max([len(paths_list) for paths_list in kti_vector]) > MAX_PATHS:
        #    print("Number of full paths for a token was longer than pre-calculated and would be cut. ")

        #path_lengths_list = [len(path) for paths_list in kti_vector for path in paths_list]
        #if (0 if len(path_lengths_list) == 0 else max(path_lengths_list)) > MAX_PATH_LEN:
        #    print("Number of full paths for a token was longer than pre-calculated and would be cut. ")

        # do actual padding for sentence length
        padding_len = MAX_SENT_LEN - len(kti_vector)
        if padding_len > 0:
            kti_vector = [[] for _ in range(padding_len)] + kti_vector
        if padding_len < 0:
            kti_vector = kti_vector[-MAX_SENT_LEN:]

        for i, full_paths in enumerate(kti_vector):
            # do padding for paths per token
            padding_len = MAX_PATHS - len(full_paths)
            if padding_len > 0:
                full_paths = [[] for _ in range(padding_len)] + full_paths
            if padding_len < 0:
                full_paths = full_paths[-MAX_PATHS:]

            # do padding for path lengths
            for j, full_path in enumerate(copy.deepcopy(full_paths)):
                padding_len = MAX_PATH_LEN - len(full_path)
                if padding_len > 0:
                    full_path = [0]*padding_len + full_path
                if padding_len < 0:
                    full_path = full_path[:padding_len] # for paths lengths, we truncate at the end
                full_paths[j] = full_path
            kti_vector[i] = full_paths
        all_full_to_indices[kti_vec_index] = kti_vector

    return np.array(all_full_to_indices)

def update_max_lengths_dict(dictionary, sample_indices, max_len_distr_dict, prefix="max_path", setting="topic_sent"):
    max_paths = []
    max_paths_len = []

    if prefix == "max_sent_len":
        max_len_distr_dict['per_sent'][len(sample_indices)] += 1
        dictionary[prefix] = max(max_len_distr_dict['per_sent'].keys())
        return

    for concept_list in sample_indices:
        max_paths.append(len(concept_list))
        if prefix == "max_path":  # only makes sense for paths
            max_paths_len.extend([len(c) for c in concept_list])

    if max(max_paths) > dictionary[prefix + 's_' + setting]:
        dictionary[prefix + 's_' + setting] = max(max_paths)

    if prefix == "max_path":  # only makes sense for paths
        if len(max_paths_len) > 0 and max(max_paths_len) > dictionary[prefix + '_len_' + setting]:
            dictionary[prefix + '_len_' + setting] = max(max_paths_len)

    # update distribution dict
    if prefix == "max_path": # for full paths
        max_len_distr_dict['max_path_len']['per_sent'][0 if len(max_paths_len) == 0 else str(max(max_paths_len))] += 1
        #max_len_distr_dict['max_path_len']['per_token'][0 if len(max_paths_len) == 0 else str(max(max_paths_len))] += 1
        max_len_distr_dict['max_paths']['per_sent'][str(max(max_paths))] += 1
        #max_len_distr_dict['max_paths']['per_token'][str(max_paths.count(max(max_paths)))] += 1
    else:
        max_len_distr_dict['per_sent'][str(max(max_paths))] += 1
        #max_len_distr_dict['per_token'][str(max_paths.count(max(max_paths)))] += 1

def restrict_max_lengths_dict(max_lengths_dict, max_concepts, max_paths, max_path_len, max_sent_len):

    # restrict all sentence lengths
    for key, value in max_lengths_dict['sentences'].items():
        if value > max_sent_len:
            max_lengths_dict['sentences'][key] = max_sent_len

    # restrict all shallow paths lengths (sent, topic, topic_sent)
    for key, value in max_lengths_dict['shallow_paths'].items():
        if value > max_concepts:
            max_lengths_dict['shallow_paths'][key] = max_concepts

    # restrict all shallow knowledge lengths (sent, topic, topic_sent)
    for key, value in max_lengths_dict['shallow_knowledge'].items():
        if value > max_concepts:
            max_lengths_dict['shallow_knowledge'][key] = max_concepts

    # restrict all full path lengths
    for key, value in max_lengths_dict['full_paths'].items():
        if "max_paths" in key and max_lengths_dict['full_paths'][key] > max_paths:
            max_lengths_dict['full_paths'][key] = max_paths
        if "max_path_len" in key and max_lengths_dict['full_paths'][key] > max_path_len:
            max_lengths_dict['full_paths'][key] = max_path_len

def save_max_len_distributions(distr_dict, PROCESSED_DIR, granularity="per_sent"):
    out_file = ""

    for key, value in distr_dict.items():
        if not "full_paths" in key:
            sorted_tuples = sorted([(int(length), amount) for length, amount in value[granularity].items()], key=lambda x: x[0])
            out_file += str(key) + "\n" + "\n".join(str(length)+"\t"+str(amount) for length, amount in sorted_tuples) + "\n\n"
        else:
            for k, v in value.items():
                sorted_tuples = sorted([(int(length), amount) for length, amount in v[granularity].items()], key=lambda x: x[0])
                out_file += str(key+": "+k) + "\n" + "\n".join(str(length) + "\t" + str(amount) for length, amount in sorted_tuples) + "\n\n"
    with open(PROCESSED_DIR+"max_len_distributions_"+granularity+".txt", "w") as f:
        f.write(out_file)
    print("Save length distributions of concepts and paths to " + PROCESSED_DIR+"max_len_distributions_"+granularity+".txt")

def get_we_vector_for_token(token, word_vecs_we, word_vecs_min, word_vecs_max, debug=True):
    try:
        return word_vecs_we.get_vector(token), True
    except KeyError as ke:
        if debug == True:
            print(ke)
        return np.random.uniform(word_vecs_min, word_vecs_max, word_vecs_we.vector_size), False

def load_wikiId_to_vec(path):
    entityId_to_vecindex = {}
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # skip the headers
        for row in reader:
            entityId_to_vecindex[row[0]] = row[1]
    return entityId_to_vecindex

'''
TODO CON

currently:
1. look Qxx in ent2id and Pxx in rel2id if not init randomly
2. look entity id in ent2vec and relation id in rel2vec
future has differnt types of entities:
- wiki entity (Q123) => search in ent2id
- wiki relation (P123) => search in rel2id
- openie entity (school uniforms) => get closest ent and search in ent2id
- openie relation (finds) => get closest rel and search in rel2id
- WIKIFIERED relation (WIKIFIERED) => random
- OPENIE relation (=openie=) => random

further rules:
- EOe => kill Oe
- eOE => kill eO
- ..T.. => kill something


:param entId2vecId: dic of entity2id.txt
:param relId2vecID: dic of relation2id.txt
:param vecId2entvec: dic of entity2vec.vec
:param vecId2relvec: dic of relation2vec.vec
:param wikilabel: Obama|_e
:param wikiid: Q123
:param ent_tag: |_e
:param rel_tag: |_r
:param min/max/dim: values for random init
:return: (result vector, bool if found or random)
'''
def get_kge_vector_for_entity(entityId_to_vecindex, relationId_to_vecindex, vecindex_to_entvec, vecindex_to_relvec,
                              wikiLabel, wikiId, vocab_tag_ent, vocab_tag_open, vocab_tag_wiki, vocab_tag_rel, 
                              vocab_tag_ento, vocab_tag_relo, vocab_tag_tok, word_vecs_min, word_vecs_max, dim):
    try:
        if wikiLabel.endswith(vocab_tag_ent):
            vecindex = int(entityId_to_vecindex.get(wikiId, None))
        elif wikiLabel.endswith(vocab_tag_rel):
            vecindex = int(relationId_to_vecindex.get(wikiId, None))
        else:
            raise Exception
    except TypeError as te:
        print('Wiki-QID "' + str(wikiId) + '" not found in the KGEs.')
        vecindex = None
    except KeyError as ke:
        print('No Wiki-QID found for label "' + str(ke) + '" (that is how it is supposed to be)')
        vecindex = None
    except Exception as ex:
        print('Problem with ent/rel tag for label "' + str(ex) + '" detected')
        vecindex = None

    if vecindex == None:
        return np.random.uniform(word_vecs_min, word_vecs_max, dim), False
    else:
        if wikiLabel.endswith(vocab_tag_ent):
            return vecindex_to_entvec[vecindex], True
        elif wikiLabel.endswith(vocab_tag_rel):
            return vecindex_to_relvec[vecindex], True
        else:
            raise Exception

def get_kge_vector_for_entity_w2v(wikiLabel, wikiId, entity_w2v, relation_w2v, vocab_tag_ent, vocab_tag_open, 
                                vocab_tag_wiki, vocab_tag_rel, vocab_tag_ento, vocab_tag_relo, vocab_tag_tok,
                                word_vecs_min, word_vecs_max, debug=True):

    try:
        if wikiLabel.endswith(vocab_tag_ent):
            return entity_w2v.get_vector(wikiId), True
        elif wikiLabel.endswith(vocab_tag_rel):
            if relation_w2v == None:  # there might be no vectors for relations
                return np.random.uniform(word_vecs_min, word_vecs_max, entity_w2v.vector_size), False
            return relation_w2v.get_vector(wikiId), True
        else:
            raise Exception
    except KeyError as ke:
        if debug == True:
            print(ke)
        return np.random.uniform(word_vecs_min, word_vecs_max, entity_w2v.vector_size), False
    except Exception as ex:
        print('Problem with ent/rel tag for label "' + str(ex) + '" detected')
        return np.random.uniform(word_vecs_min, word_vecs_max, entity_w2v.vector_size), False