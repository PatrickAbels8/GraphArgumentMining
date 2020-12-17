import json
from helpers.classification import generate_features
import numpy as np
import os
import copy
from models import baseline_bilstm_cos, baseline_clstm, baseline_bilstm, baseline_evlstm_inner_att, \
    baseline_evlstm_inner_topic_att, \
    baseline_evlstm_full_path_inner_att, baseline_evclstm_inner_att, baseline_evclstm_inner_topic_att, \
    baseline_evlstm_full_path_inner_topic_att, baseline_evlstm_full_path_avg_inner_att,\
    baseline_evlstm_full_path_stacked_inner_att, baseline_bilstm_bert, baseline_evlstm_inner_att_bert, \
    baseline_bilstm_only_knowledge, baseline_bilstm_bert_sent_knowledge_concat, baseline_mlp_bert,\
    baseline_bilstm_bert_sent_knowledge_concat_dense, baseline_evlstm_inner_att_sent_and_knowledge_bert
#from models.tensorflow import baseline_evlstm_inner_topic_att, baseline_evlstm_inner_att_cos, \
#    baseline_evlstm_inner_topic_att_cos, baseline_evlstm_inner_att, baseline_noevlstm, baseline_evlstm_inner_4dim_att, \
#    baseline_evlstm
from helpers.classification.scoring import add_scores, add_topic_avg_key, add_seed_avg_key, save_json, get_model_file_suffix
from utils.helper import get_time

def load_splits_UKPSententialArgMin(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, model_settings):

    def get_y_data(topic, SAMPLES_DIR, label_setup, dataset):
        # loads y data from the JSON data files that hold the knowledge as strings. Converts depending on setup and to 1-hot
        y = []
        with open(SAMPLES_DIR + "result_"+topic+"_nx.json", "r") as f:
            sample_data = json.load(f)
            for sample in sample_data["samples"]:
                y.append(generate_features.get_label_index_from_setup(sample["label"], label_setup, dataset))
        y = np.eye(np.max(y) + 1, dtype=int)[y] # to one hot
        return y

    # set data path
    if model_settings['kg_embeddings'][0] == "BERT_KNOWLEDGE_SENT":
        PROCESSED_KNOWLEDGE_DIR = PROCESSED_DIR+model_settings["knowledge_config"]+"_bert_sent_emb/"
    else:
        PROCESSED_KNOWLEDGE_DIR = PROCESSED_DIR+model_settings["knowledge_config"]+"/"


    # load json that holds the splits for all topics
    with open(PROCESSED_DIR + "preset_indices_split.json", "r") as f:
        split_dict = json.load(f)

    # if BERT embeddings are used, take the train data prepared for that
    if "BERT" in model_settings['word_embeddings'][0]:
        PROCESSED_DIR = PROCESSED_SENTENCES_BERT_DIR

    # get paths to topic data (train/dev/test)
    topic_data = [file for file in os.listdir(PROCESSED_DIR) if os.path.isfile(PROCESSED_DIR+file) \
                  and file.endswith("_X.npy")]

    train_data_dict = {}
    for data_X_file in topic_data:
        topic_name = data_X_file.split("_X.npy")[0]
        data_kX_file = topic_name + "_kX.npy"

        # load X and y feature vectors
        X = np.load(PROCESSED_DIR+data_X_file)
        kX = np.load(PROCESSED_KNOWLEDGE_DIR+data_kX_file)
        y = get_y_data(topic_name, SAMPLES_DIR, model_settings["label_setup"], model_settings["dataset"])

        assert len(X) == len(kX) == len(y), "Different lengths between data in X, knowledge of X (kX) and labels y detected!"

        train_data_dict[topic_name] = {
            # train data
            "X_train": X.take(split_dict[topic_name]["train_indices"], axis=0),
            "kX_train": kX.take(split_dict[topic_name]["train_indices"], axis=0),
            "X_topic_train": [topic_name] * len(split_dict[topic_name]["train_indices"]),
            "y_train": y.take(split_dict[topic_name]["train_indices"], axis=0),

            # dev data
            "X_dev": X.take(split_dict[topic_name]["dev_indices"], axis=0),
            "kX_dev": kX.take(split_dict[topic_name]["dev_indices"], axis=0),
            "X_topic_dev": [topic_name] * len(split_dict[topic_name]["dev_indices"]),
            "y_dev": y.take(split_dict[topic_name]["dev_indices"], axis=0),

            # test data
            "X_test": X.take(split_dict[topic_name]["test_indices"], axis=0),
            "kX_test": kX.take(split_dict[topic_name]["test_indices"], axis=0),
            "X_topic_test": [topic_name] * len(split_dict[topic_name]["test_indices"]),
            "y_test": y.take(split_dict[topic_name]["test_indices"], axis=0),
        }

    return train_data_dict

def load_splits_IBMDebaterEvidenceSentences(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, model_settings):

    def get_y_data(topic, SAMPLES_DIR, label_setup, dataset):
        # loads y data from the JSON data files that hold the knowledge as strings. Converts depending on setup and to 1-hot
        y = []
        with open(SAMPLES_DIR + "result_"+topic+"_nx.json", "r") as f:
            sample_data = json.load(f)
            for sample in sample_data["samples"]:
                y.append(generate_features.get_label_index_from_setup(sample["label"], label_setup, dataset))
        y = np.eye(np.max(y) + 1, dtype=int)[y] # to one hot
        return y

    # set data path
    PROCESSED_KNOWLEDGE_DIR = PROCESSED_DIR+model_settings["knowledge_config"]+"/"

    # load json that holds the splits for all topics
    with open(PROCESSED_DIR + "preset_indices_split.json", "r") as f:
        split_dict = json.load(f)

    # if BERT embeddings are used, take the train data prepared for that
    if model_settings['word_embeddings'][0] == "BERT":
        PROCESSED_DIR = PROCESSED_SENTENCES_BERT_DIR

    # get paths to topic data (train/dev/test)
    topic_data = [file for file in os.listdir(PROCESSED_DIR) if os.path.isfile(PROCESSED_DIR+file) \
                  and file.endswith("_X.npy")]

    train_data_dict = {}
    for data_X_file in topic_data:
        topic_name = data_X_file.split("_X.npy")[0]
        data_kX_file = topic_name + "_kX.npy"

        # load X and y feature vectors
        X = np.load(PROCESSED_DIR+data_X_file)
        kX = np.load(PROCESSED_KNOWLEDGE_DIR+data_kX_file)
        y = get_y_data(topic_name, SAMPLES_DIR, model_settings["label_setup"], model_settings["dataset"])

        assert len(X) == len(kX) == len(y), "Different lengths between data in X, knowledge of X (kX) and labels y detected!"

        train_data_dict[topic_name] = {
            # train data
            "X_train": X.take(split_dict[topic_name]["train_indices"], axis=0),
            "kX_train": kX.take(split_dict[topic_name]["train_indices"], axis=0),
            "X_topic_train": [topic_name] * len(split_dict[topic_name]["train_indices"]),
            "y_train": y.take(split_dict[topic_name]["train_indices"], axis=0),

            # dev data
            "X_dev": X.take(split_dict[topic_name]["test_indices"], axis=0),
            "kX_dev": kX.take(split_dict[topic_name]["test_indices"], axis=0),
            "X_topic_dev": [topic_name] * len(split_dict[topic_name]["test_indices"]),
            "y_dev": y.take(split_dict[topic_name]["test_indices"], axis=0),

            # test data
            "X_test": X.take(split_dict[topic_name]["test_indices"], axis=0),
            "kX_test": kX.take(split_dict[topic_name]["test_indices"], axis=0),
            "X_topic_test": [topic_name] * len(split_dict[topic_name]["test_indices"]),
            "y_test": y.take(split_dict[topic_name]["test_indices"], axis=0),
        }

    return train_data_dict

def load_splits(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, model_settings):
    if model_settings['dataset'] == "UKPSententialArgMin":
        return load_splits_UKPSententialArgMin(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, model_settings)
    elif model_settings['dataset'] == "IBMDebaterEvidenceSentences":
        return load_splits_IBMDebaterEvidenceSentences(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, model_settings)
    else:
        raise Exception("Dataset not recognized!")

def train_procedure(train_data_dict, model_fct_dict, model_name, RESULTS_DIR, PROCESSED_DIR,
                    model_settings={}, predict_test=False):

    train_fct = model_fct_dict[model_name]

    results_DEV = {}
    results_TEST = {}
    MODEL_FOLDER = RESULTS_DIR + model_name + "/"
    num_seeds = model_settings.get("num_seeds", 10)
    print("====================== Start " +model_settings["train_setup"]+ " experiments for " + model_settings['model'] + " ======================")

    for seed in range(num_seeds):
        model_settings['current_seed'] = seed
        seed = str(seed)
        print("=> Start experiments for seed " + seed)
        results_DEV[str(seed)] = {}
        results_TEST[str(seed)] = {}
        SEED_FOLDER = MODEL_FOLDER + "seed_" + seed + "/"
        model_file_suffix = get_model_file_suffix(model_settings)
        dev_result_file_name = "DEV_result_"+model_file_suffix
        test_result_file_name = "TEST_result_"+model_file_suffix
        model_settings['model_file_suffix'] = model_file_suffix
        placeholder_topic = "IBM_all" if model_settings['dataset'] == "IBMDebaterEvidenceSentences" else None

        # Create needed folder(s)
        try:
            os.makedirs(SEED_FOLDER)
        except IOError as e:
            print(e)

        # Special setup of UKP corpus
        if model_settings['dataset'] == "UKPSententialArgMin":
            #print("Data dict items: ", train_data_dict.items())
            tdd_dict = list(train_data_dict.items())
            # tdd_dict.reverse()
            tdd_dict[0], tdd_dict[4] = tdd_dict[4], tdd_dict[0]
            for topic, data in tdd_dict:

            # for topic, data in train_data_dict.items():  # {t:d for t,d in train_data_dict.items() if "school" in t}.items()
                print("==> Current topic: " + str(topic))
                if model_settings["train_setup"] == "cross_domain":

                    # get dev and test data just from current topic
                    cross_domain_data = copy.deepcopy(data)
                    cross_domain_data.pop('X_train')
                    cross_domain_data.pop('X_topic_train')
                    cross_domain_data.pop('kX_train')
                    cross_domain_data.pop('y_train')
                    cross_domain_data.pop('X_dev')
                    cross_domain_data.pop('X_topic_dev')
                    cross_domain_data.pop('kX_dev')
                    cross_domain_data.pop('y_dev')

                    # get train data by taking the data of all topics except current one
                    count_t = 0
                    for count_topic, count_data in train_data_dict.items():
                        if count_topic == topic: # skip test topic data for training/dev data gen
                            continue

                        if count_t == 0:
                            cross_domain_data['X_train'] = np.copy(count_data['X_train'])
                            cross_domain_data['X_topic_train'] = count_data['X_topic_train'][:]
                            cross_domain_data['kX_train'] = np.copy(count_data['kX_train'])
                            cross_domain_data['y_train'] = np.copy(count_data['y_train'])
                            cross_domain_data['X_dev'] = np.copy(count_data['X_dev'])
                            cross_domain_data['X_topic_dev'] = count_data['X_topic_dev'][:]
                            cross_domain_data['kX_dev'] = np.copy(count_data['kX_dev'])
                            cross_domain_data['y_dev'] = np.copy(count_data['y_dev'])
                        else:
                            cross_domain_data['X_train'] = np.concatenate([cross_domain_data['X_train'], np.copy(count_data['X_train'])], axis=0)
                            cross_domain_data['kX_train'] = np.concatenate([cross_domain_data['kX_train'], np.copy(count_data['kX_train'])], axis=0)
                            cross_domain_data['X_topic_train'] = cross_domain_data['X_topic_train'] + count_data['X_topic_train'][:]
                            cross_domain_data['y_train'] = np.concatenate([cross_domain_data['y_train'], np.copy(count_data['y_train'])], axis=0)
                            cross_domain_data['X_dev'] = np.concatenate([cross_domain_data['X_dev'], np.copy(count_data['X_dev'])], axis=0)
                            cross_domain_data['X_topic_dev'] = cross_domain_data['X_topic_dev'] + count_data['X_topic_dev']
                            cross_domain_data['kX_dev'] = np.concatenate([cross_domain_data['kX_dev'], np.copy(count_data['kX_dev'])], axis=0)
                            cross_domain_data['y_dev'] = np.concatenate([cross_domain_data['y_dev'], np.copy(count_data['y_dev'])], axis=0)

                        count_t += 1
                    # run classifier
                    y_test_pred, y_dev_pred = train_fct(cross_domain_data, topic, PROCESSED_DIR, SEED_FOLDER,
                                                        model_settings=model_settings)

                    # calc scores
                    results_DEV[seed] = add_scores(y_test_pred, y_dev_pred, np.copy(cross_domain_data['y_test']), np.copy(cross_domain_data['y_dev']),
                                               topic, results_DEV[seed], model_settings['label_setup'], is_test=False)
                    print("f1_macro dev: " + str(results_DEV[seed][topic]["dev"]["f1_macro"]))

                    if predict_test == True:
                        results_TEST[seed] = add_scores(y_test_pred, y_dev_pred, np.copy(cross_domain_data['y_test']), np.copy(cross_domain_data['y_dev']),
                                                   topic, results_TEST[seed], model_settings['label_setup'], is_test=True)
                        print("f1_macro test: " + str(results_TEST[seed][topic]["test"]["f1_macro"]))

                elif model_settings["train_setup"] == "in_domain":
                    # run classifier
                    y_test_pred, y_dev_pred = train_fct(data, topic, PROCESSED_DIR, SEED_FOLDER,
                                                        model_settings=model_settings)

                    # calc scores
                    results_DEV[seed] = add_scores(y_test_pred, y_dev_pred, np.copy(data['y_test']), np.copy(data['y_dev']),
                                               topic, results_DEV[seed], model_settings['label_setup'], is_test=False)
                    print("f1_macro dev: " + str(results_DEV[seed][topic]["dev"]["f1_macro"]))

                    if predict_test == True:
                        results_TEST[seed] = add_scores(y_test_pred, y_dev_pred, np.copy(data['y_test']), np.copy(data['y_dev']),
                                                   topic, results_TEST[seed], model_settings['label_setup'], is_test=True)
                        print("f1_macro test: " + str(results_TEST[seed][topic]["test"]["f1_macro"]))
                else:
                    raise Exception("Train_setup not recognized!")
        elif model_settings['dataset'] == "IBMDebaterEvidenceSentences":
            cumulated_data_dict = {}
            for key_td in train_data_dict[list(train_data_dict.keys())[0]]:
                cumulated_data_dict[key_td] = [] # add keys x_train, kX_train, ...
            for _, topic_data_td in train_data_dict.items():
                for key_td, key_data_td in topic_data_td.items():
                    if len(key_data_td) > 0:
                        cumulated_data_dict[key_td].append(key_data_td)
            for key_td, key_data_td in cumulated_data_dict.items():
                if "topic" in key_td:
                    cumulated_data_dict[key_td] = [item for sublist in key_data_td for item in sublist]
                else:
                    cumulated_data_dict[key_td] = np.vstack(key_data_td)

            y_test_pred, y_dev_pred = train_fct(cumulated_data_dict, placeholder_topic, PROCESSED_DIR, SEED_FOLDER,
                                                model_settings=model_settings)

            # calc scores
            results_DEV[seed] = add_scores(y_test_pred, y_dev_pred, np.copy(cumulated_data_dict['y_test']), np.copy(cumulated_data_dict['y_dev']),
                                           placeholder_topic, results_DEV[seed], model_settings['label_setup'], is_test=False)
            print("f1_macro dev: " + str(results_DEV[seed][placeholder_topic]["dev"]["f1_macro"]))

            if predict_test == True:
                results_TEST[seed] = add_scores(y_test_pred, y_dev_pred, np.copy(cumulated_data_dict['y_test']),
                                                np.copy(cumulated_data_dict['y_dev']),
                                                placeholder_topic, results_TEST[seed], model_settings['label_setup'], is_test=True)
                print("f1_macro test: " + str(results_TEST[seed][placeholder_topic]["test"]["f1_macro"]))

        # add avg over all scores
        results_DEV[seed] = add_topic_avg_key(results_DEV[seed], is_test=False)
        if predict_test == True:
            results_TEST[seed] = add_topic_avg_key(results_TEST[seed], is_test=True)

        # save results to file for the seeds classified so far (checkpoint)
        save_json(MODEL_FOLDER, dev_result_file_name, results_DEV)
        if predict_test == True:
            save_json(MODEL_FOLDER, test_result_file_name, results_TEST)


    # add results over all seeds
    results_DEV = add_seed_avg_key(results_DEV, placeholder_topic, is_test=False)
    if predict_test == True:
        results_TEST = add_seed_avg_key(results_TEST, placeholder_topic, is_test=True)

    # add config of the model
    model_settings.pop("current_seed")
    model_settings["training_end"] = get_time()
    results_DEV['config'] = model_settings
    if predict_test == True:
        results_TEST['config'] = model_settings

    # save results to file
    save_json(MODEL_FOLDER, dev_result_file_name, results_DEV)
    if predict_test == True:
        save_json(MODEL_FOLDER, test_result_file_name, results_TEST)


def train_clf(model, SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, RESULTS_DIR, model_settings={}, predict_test=False):

    train_model = {
        "bilstm_cos": baseline_bilstm_cos.train_model,
        "bilstm": baseline_bilstm.train_model,
        "bilstm_bert": baseline_bilstm_bert.train_model,
        "bilstm_bert_sent_knowledge_concat": baseline_bilstm_bert_sent_knowledge_concat.train_model,
        "bilstm_bert_sent_knowledge_concat_dense": baseline_bilstm_bert_sent_knowledge_concat_dense.train_model,
        "mlp_bert": baseline_mlp_bert.train_model,
        "bilstm_only_knowledge": baseline_bilstm_only_knowledge.train_model,
       # "EvLSTM": baseline_evlstm.train_model,
        "EvLSTM_inner_att": baseline_evlstm_inner_att.train_model,
        "EvLSTM_inner_att_keras": baseline_evlstm_inner_att.train_model,
        "EvLSTM_inner_att_bert_keras": baseline_evlstm_inner_att_bert.train_model,
        "EvLSTM_inner_att_sent_and_knowledge_bert_keras": baseline_evlstm_inner_att_sent_and_knowledge_bert.train_model,
        "EvCLSTM_inner_att_keras": baseline_evclstm_inner_att.train_model,
        "EvCLSTM_inner_topic_att_keras": baseline_evclstm_inner_topic_att.train_model,
        "EvLSTM_inner_topic_att_keras": baseline_evlstm_inner_topic_att.train_model,
        "EvLSTM_full_path_inner_att_keras": baseline_evlstm_full_path_inner_att.train_model,
        "EvLSTM_full_path_inner_topic_att_keras": baseline_evlstm_full_path_inner_topic_att.train_model,
        "EvLSTM_full_path_avg_inner_att_keras": baseline_evlstm_full_path_avg_inner_att.train_model,
        "EvLSTM_full_path_stacked_inner_att_keras": baseline_evlstm_full_path_stacked_inner_att.train_model,
        "clstm": baseline_clstm.train_model,
       # "NoEvLSTM": baseline_noevlstm.train_model,
        "EvLSTM_inner_topic_att": baseline_evlstm_inner_topic_att.train_model,
       # "EvLSTM_inner_topic_att_cos": baseline_evlstm_inner_topic_att_cos.train_model,
       # "EvLSTM_inner_att_cos": baseline_evlstm_inner_att_cos.train_model,
       # "EvLSTM_inner_4dim_att": baseline_evlstm_inner_4dim_att.train_model,

    }

    train_data_dict = load_splits(SAMPLES_DIR, PROCESSED_DIR, PROCESSED_SENTENCES_BERT_DIR, model_settings)
    print("Loaded train_data_dict:", train_data_dict)
    return train_procedure(train_data_dict, train_model, model, RESULTS_DIR, PROCESSED_DIR,
                           model_settings=model_settings, predict_test=predict_test)