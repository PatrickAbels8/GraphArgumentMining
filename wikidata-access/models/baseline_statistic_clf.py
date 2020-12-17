import os
import json
import numpy as np
from helpers.classification import generate_features
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def get_features(sentence, paths_to_topic, paths_between_entities, vocab, feature_list):

    """
            feature_list = ['bow', 'sent_len', 'tokens_linked_within_sent', 'tokens_linked_to_topic',
                        'sents_with_paths_between_entities', 'sents_with_path_to_topic',
                        'include_only_topic', 'include_topic']
    :param sentence:
    :param paths_to_topic:
    :param feature_list:
    :return:
    """

    ngram_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, lowercase=True, vocabulary=vocab)
    all_features = {
        "bow": ngram_vectorizer.transform([sentence]).toarray()[0].tolist(), # bag of words / bow
        "sent_len": len(nltk.word_tokenize(sentence)), #sent_len
        "tokens_linked_within_sent": generate_features.get_no_linked_token(paths_between_entities), # tokens_linked_within_sent
        "tokens_linked_to_topic": generate_features.get_no_linked_token(paths_to_topic), # tokens_linked_to_topic
        "sents_with_paths_between_entities": (1 if len(paths_between_entities) > 0 else 0), # sents_with_paths_between_entities
        "sents_with_path_to_topic": (1 if len(paths_to_topic) > 0 else 0), # sents_with_path_to_topic
        "include_only_topic": (1 if max(generate_features.get_path_lengths(paths_to_topic)) == 1 else 0),  # include_only_topic
        "include_topic": (1 if 1 in generate_features.get_path_lengths(paths_to_topic) else 0) # include_topic
    }

    # use only features that are needed
    features = []
    for feat in feature_list:
        if isinstance(all_features[feat], list):
            features.extend(all_features[feat])
        else:
            features.append(all_features[feat])

    return features

def create_features(label_setup, feature_list, SAMPLES_DIR, PROCESSED_DIR):
    # load data and prepare data
    topic_dict = {}
    gold_files = [file for file in os.listdir(SAMPLES_DIR) if os.path.isfile(SAMPLES_DIR+file)]
    for data_file in gold_files:
        with open(SAMPLES_DIR+data_file, "r") as f:
            topic_name = data_file.split("result_")[1].split("_nx")[0]

            # load data for topic and get number of samples for the topic
            topic_data = json.load(f)

            bow_sents = [] # only add training data
            sents = []
            labels = []
            paths_to_topic = []
            paths_between_entities = []

            for sample in topic_data["samples"]:
                sents.append(sample["sentence"])
                labels.append(sample["label"])
                paths_to_topic.append(sample["paths_to_topic"])
                paths_between_entities.append(sample["paths_between_entities"])
                if sample["set"] == "train":
                    bow_sents.append(sample["sentence"])

            # get vocab => since we take all words into account, no need to use TfIDF
            ngram_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, lowercase=True)  # lowercase=False, decode_error='ignore', encoding='utf',
            ngram_vectorizer.fit_transform(bow_sents)

            topic_dict[topic_name] = {
                "sentences": sents,
                "paths_to_topic": paths_to_topic,
                "paths_between_entities": paths_between_entities,
                "labels": labels,
                "vocab": ngram_vectorizer.vocabulary_
            }

    # create and save actual training/dev/test data
    for topic, values in topic_dict.items():
        X_data = []
        y_data = []

        # create features and training data
        print("Generate training data of topic {0} for baseline statistic classifier".format(topic))
        for i in range(len(values["sentences"])):
            X_data.append(get_features(values["sentences"][i],
                                       values["paths_to_topic"][i],
                                       values["paths_between_entities"][i],
                                       values["vocab"],
                                       feature_list))
            y_data.append(generate_features.get_label_index_from_setup(values["labels"][i], label_setup))

        # save topic_X, topic_y
        np.save(PROCESSED_DIR + topic + "_X.npy", X_data)
        np.save(PROCESSED_DIR + topic + "_y.npy", y_data)

    
def train_model(SAMPLES_DIR, label_setup, feature_list):
    RESULT_DIR = SAMPLES_DIR+"/model_runs/statistic_clf/"
    #three_label = ['Argument_against', 'Argument_for', 'NoArgument']
    #two_label = ['Argument', 'NoArgument']

    # Create needed folder(s)
    try:
        os.makedirs(RESULT_DIR)
    except IOError as e:
        print(e)

    # create features
    create_features(label_setup, feature_list, SAMPLES_DIR, RESULT_DIR)

    # start classification
    # load split info file
    with open(SAMPLES_DIR + "processed/preset_indices_split.json", "r") as f:
        split_dict = json.load(f)

    topic_data = [file for file in os.listdir(RESULT_DIR) if os.path.isfile(RESULT_DIR+file) and file.endswith("_X.npy")]
    f1_macro_values = []
    for data_X_file in topic_data:
        data_y_file = data_X_file.split("_X.npy")[0] + "_y.npy"

        # load X and y feature vectors
        X = np.load(RESULT_DIR+data_X_file)
        y = np.load(RESULT_DIR+data_y_file)

        topic_name = data_X_file.split("_X.npy")[0]
        X_train = X.take(split_dict[topic_name]["train_indices"], axis=0)
        y_train_gold = y.take(split_dict[topic_name]["train_indices"], axis=0)
        X_dev = X.take(split_dict[topic_name]["test_indices"], axis=0)
        y_dev_gold = y.take(split_dict[topic_name]["test_indices"], axis=0)

        lr = LogisticRegression(random_state=0, class_weight="balanced")
        #lr = MLPClassifier(random_state=0, verbose=1, max_iter=10, hidden_layer_sizes=(100,))
        lr.fit(X_train, y_train_gold)
        data_y_pred = lr.predict(X_dev)

        f1 = f1_score(y_dev_gold, data_y_pred, average='macro')
        f1_macro_values.append(f1)

    print("F1 macro (in-topic, 2-lbl): "+str(np.average(f1_macro_values)))


    return None