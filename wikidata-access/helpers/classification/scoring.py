from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import json
from collections import defaultdict
import ast
from .generate_features import label_setups

def to_binary_labels(y_data):
    return [np.argmax(y) for y in y_data]

def get_scores_dict(y_gold, y_pred, label_setup):
    # Get several scores (f1,prec,rec, all classwise) and return them in a dictionary

    def merge_stance_classes(y_pred, y_gold):
        # merge pro/con class for three_label setup, if precision/recall/f1 for the whole Arg class is wanted,
        # in order to view them as a single class

        y_gold_combined = []
        y_pred_combined = []

        for i in range(len(y_gold)):
            y_gold_combined.append(0 if y_gold[i] == 0 or y_gold[i] == 1 else 1)
            y_pred_combined.append(0 if y_pred[i] == 0 or y_pred[i] == 1 else 1)

        return y_pred_combined, y_gold_combined

    #"three_label": ['Argument_against', 'Argument_for', 'NoArgument'],
    #"two_label": ['Argument', 'NoArgument'],
    divisor = 3.0 if label_setup == "three_label" else 2.0

    # get f1, rec, prec scores for predictions
    f1_clw = f1_score(y_gold, y_pred, average=None)
    rec_clw = recall_score(y_gold, y_pred, average=None)
    pre_clw = precision_score(y_gold, y_pred, average=None)

    # if three_label, merge pro/con class to get a combined Argument class
    if label_setup == "three_label":
        y_pred_combined, y_gold_combined = merge_stance_classes(y_pred, y_gold)
        f1_clw_combined = f1_score(y_gold_combined, y_pred_combined, average=None)
        rec_clw_combined = recall_score(y_gold_combined, y_pred_combined, average=None)
        pre_clw_combined = precision_score(y_gold_combined, y_pred_combined, average=None)

    return {
        "f1_macro": sum(f1_clw) / divisor,
        "rec_macro": sum(rec_clw) / divisor,
        "pre_macro": sum(pre_clw) / divisor,

        "accuracy": accuracy_score(y_gold, y_pred),

        "f1_arg": f1_clw[0] if label_setup == "two_label" else f1_clw_combined[0],  # if 3lbl, convert to 2lbl
        "f1_arg_con": f1_clw[0] if label_setup == "three_label" else -1,  # only three lbl
        "f1_arg_pro": f1_clw[1] if label_setup == "three_label" else -1,  # only three lbl
        "f1_noArg": f1_clw[1] if label_setup == "two_label" else f1_clw[2],

        "pre_arg": pre_clw[0] if label_setup == "two_label" else pre_clw_combined[0], # if 3lbl, convert to 2lbl
        "pre_arg_con": pre_clw[0] if label_setup == "three_label" else -1,  # only three lbl
        "pre_arg_pro": pre_clw[1] if label_setup == "three_label" else -1,  # only three lbl
        "pre_noArg": pre_clw[1] if label_setup == "two_label" else pre_clw[2],

        "rec_arg": rec_clw[0] if label_setup == "two_label" else rec_clw_combined[0], # if 3lbl, convert to 2lbl
        "rec_arg_con": rec_clw[0] if label_setup == "three_label" else -1,  # only three lbl
        "rec_arg_pro": rec_clw[1] if label_setup == "three_label" else -1,  # only three lbl
        "rec_noArg": rec_clw[1] if label_setup == "two_label" else rec_clw[2]
    }

def add_scores(y_test_pred, y_dev_pred, y_test_gold, y_dev_gold, topic, results, label_setup, gold_to_binary=True, is_test=False):
    # Get several scores for test and dev set and returns them in a dict {topic: { dev: {...}, test: {...} } }

    if gold_to_binary == True:
        y_test_gold = to_binary_labels(y_test_gold)
        y_dev_gold = to_binary_labels(y_dev_gold)
        # pre_clw_test_combined[1] == pre_clw_test[2]

    results[topic] = {
        "model_out": {
            "test": {
                "y_gold": str(y_test_gold),
                "y_pred": str(y_test_pred)
            },
            "dev": {
                "y_gold": str(y_dev_gold),
                "y_pred": str(y_dev_pred)
                }
        }
    }

    if is_test == True:
        results[topic]["test"] = get_scores_dict(y_test_gold, y_test_pred, label_setup)
    else:
        results[topic]["dev"] = get_scores_dict(y_dev_gold, y_dev_pred, label_setup)

    return results

def add_topic_avg_key(dictionary, is_test=False):
    set_key = "dev" if is_test == False else "test"

    # initilize dict with all-key
    init_keys = dictionary[list(dictionary.keys())[0]][set_key].keys()
    dictionary["all"] = {set_key: defaultdict(list)}

    # get the scores for the topics and add them to a list in the all-dict
    for topic, values in dictionary.items():
        if topic != "all":
            for k in init_keys:
                dictionary["all"][set_key][k].append(values[set_key][k])

    # calculate the stdev, add them to the dict, and the replace the list of values with the final averaged result for the metric
    for k in init_keys:
        dictionary["all"][set_key][k+"_stdev"] = np.std(dictionary["all"][set_key][k])
        dictionary["all"][set_key][k] = np.average(dictionary["all"][set_key][k])

    return dictionary

def add_micro_avg_accuracy(dictionary, set_key, placeholder_topic):
    # calculates the micro average accuracy over all seeds for the IBM Debater Evidence Corpus
    y_gold, y_pred = [], []
    for seed, values in dictionary.items():
        y_gold.extend([int(x) for x in ast.literal_eval(values[placeholder_topic]['model_out'][set_key]["y_gold"])])
        y_pred.extend([int(x) for x in ast.literal_eval(values[placeholder_topic]['model_out'][set_key]["y_pred"])])
    return accuracy_score(y_gold, y_pred)

def add_seed_avg_key(dictionary, placeholder_topic, is_test=False):
    set_key = "dev" if is_test == False else "test"

    # initilize dict with all-key
    temp_seed_dict = dictionary[list(dictionary.keys())[0]]
    init_keys = temp_seed_dict[list(temp_seed_dict.keys())[0]][set_key].keys()
    avg_dict = {"all_seed_avg": {set_key: {"topics": defaultdict(list)}} }
    for k in init_keys: # add all metrics
        avg_dict["all_seed_avg"][set_key][k] = []

    # todo add rdm divisor and multiply test values with it? at least f1, pre, rec?
    # get the scores for the topics and add them to a list in the all-dict
    for seed, values in dictionary.items():
        for k in init_keys: # averaged macro scores over all seeds
            avg_dict["all_seed_avg"][set_key][k].append(values["all"][set_key][k])

        for topic in temp_seed_dict.keys(): # topic-wise average f1 scores over all seeds
            if topic == "all":
                continue
            avg_dict["all_seed_avg"][set_key]["topics"][topic+"_f1_macro"].append(values[topic][set_key]["f1_macro"])

    # calculate the stdev, add them to the dict, and the replace the list of values with the final averaged result for the metric
    for k in list(avg_dict["all_seed_avg"][set_key].keys()):
        if k == "topics":
            continue

        avg_dict["all_seed_avg"][set_key][k+"_stdev"] = np.std(avg_dict["all_seed_avg"][set_key][k])
        avg_dict["all_seed_avg"][set_key][k] = np.average(avg_dict["all_seed_avg"][set_key][k])


    # avg over topics
    for topic in temp_seed_dict.keys():
        if topic == "all":
            continue

        avg_dict["all_seed_avg"][set_key]["topics"][topic+"_f1_macro_stdev"] = np.std(avg_dict["all_seed_avg"][set_key]["topics"][topic+"_f1_macro"])
        avg_dict["all_seed_avg"][set_key]["topics"][topic+"_f1_macro"] = np.average(avg_dict["all_seed_avg"][set_key]["topics"][topic+"_f1_macro"])

    if placeholder_topic != None: # only in case of IBM corpus, add micro average accuracy score
        avg_dict["all_seed_avg"][set_key]['accuracy_micro'] = add_micro_avg_accuracy(dictionary, set_key, placeholder_topic)

    # update result dict with averages over all seeds
    dictionary.update(avg_dict)

    return dictionary

def save_json(path, filename, file):
    with open(path + filename + '.json', 'w') as outfile:
        json.dump(file, outfile, indent=4, sort_keys=True)

def get_model_file_suffix(model_settings):
    attention_size = ""
    if "attention_size" in model_settings.keys() and model_settings["attention_size"] != None:
        attention_size = "_attsize-"+ str(model_settings["attention_size"])
    return model_settings["label_setup"].replace("_", "") + "_" + model_settings["train_setup"].replace("_", "") + "_monitor-" + model_settings["monitor"] +  \
           "_do-" + str(model_settings["dropout"]) + "_lsize-" + str(model_settings["lstm_size"]) + "_bs-" + str(model_settings["batch_size"]) + \
           "_epochs-" + str(model_settings["epochs"]) + "_lr-" + str(model_settings["learning_rate"]) + \
           attention_size + "_trainemb-" + str(model_settings["train_embeddings"]) + \
           "_kl-" + model_settings["knowledge_enriched_data"] + "_kc-" + model_settings["knowledge_config"] + \
           "_we-" + model_settings["word_embeddings"][1][1:] + "_kge-" + model_settings["kg_embeddings"][1][1:]

def save_predictions(data, y_pred, y_gold, vocab_we, setup, file, gold_to_binary=True):
    # saves predictions in format: GOLD PRED SENT into file
    if gold_to_binary == True:
        y_gold_binary = to_binary_labels(y_gold)
    else:
        y_gold_binary = y_gold
    with open(file, "w", encoding="utf-8") as out_file:
        for i, X in enumerate(data):
            sent = " ".join([vocab_we[s] for s in X if s != 0])
            out_file.write(label_setups[setup][y_gold_binary[i]] + "\t" + label_setups[setup][y_pred[i]] + "\t" + sent + "\n")

