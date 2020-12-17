import itertools
import numpy as np
import matplotlib.pyplot as plt
import json
from helpers.classification.generate_features import label_setups
import ast
import copy
from sklearn.metrics import confusion_matrix

def test_matrices(cm_dict):
    temp_cm = None
    for key, value in cm_dict.items():
        if key != "all":
            if not isinstance(temp_cm, np.ndarray):
                temp_cm = copy.deepcopy(value)
            else:
                temp_cm += copy.deepcopy(value)
    return np.array_equal(cm_dict['all'], temp_cm)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrices(cm_dict, title, classes, normalize=False, cmap=plt.cm.Blues):
    rows = int((len(cm_dict.items())-1)/2)

    plt.suptitle(title, fontsize=16)
    sorted_topics = ["all"] + list(sorted([t for t in cm_dict.keys() if t != 'all']))
    for i, topic in enumerate(sorted_topics):
        plt.subplot(2, rows+1, i+1)
        plot_confusion_matrix(cm_dict[topic], classes, normalize=normalize, title=str(topic), cmap=cmap)
    plt.subplots_adjust(wspace=1.0, top=1.0, bottom=0.22)


def get_f1_from_cm(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return np.sum(f1)/len(f1)

def get_averaged_cm(result_file):
    # load gold labes and predictions of all seeds of the file and return an overaged confusion matrix of them

    with open(result_file, 'r') as f:
        loaded_results = json.load(f)


    divisor = 0
    cm_dict = {}

    for seed, seed_topics in loaded_results.items():
        if seed != "config" and seed != "all_seed_avg":
            for t_name, t_concent in seed_topics.items():
                if t_name != "all":
                    divisor += 1
                    try:
                        y_gold = [int(a) for a in ast.literal_eval(t_concent['model_out']['test']['y_gold'])]
                        y_pred = [int(a) for a in ast.literal_eval(t_concent['model_out']['test']['y_pred'])]
                    except Exception as e:
                        print(e)
                    if not "all" in cm_dict.keys():
                        cm_dict['all'] = confusion_matrix(y_gold, y_pred)
                    else:
                        cm_dict['all'] += confusion_matrix(y_gold, y_pred)

                    if not t_name in cm_dict.keys():
                        cm_dict[t_name] = confusion_matrix(y_gold, y_pred)
                    else:
                        cm_dict[t_name] += confusion_matrix(y_gold, y_pred)



    for key, value in cm_dict.items():
        value = value / (divisor if key == 'all' else len(loaded_results['0'])-1)
        value = value.astype('int32')

    # check if f1 of this matrix is similar to the one calculated by averaging the f1 scores
    print("F1 cm (micro avg): " + str(get_f1_from_cm(cm_dict['all'])) + ", F1 calculated: " + str(loaded_results['all_seed_avg']['test']['f1_macro']))

    assert test_matrices(cm_dict), 'CM in key "all" does not match the sum of topic-wise CMs'

    return cm_dict, label_setups[loaded_results['config']['label_setup']]




if __name__ == '__main__':
    BILSTM = ""
    BILSTM_BERT = ""
    BILSTM_KNOWLEDGE = ""
    BILSTM_ONLY_KNOWLEDGE = ""
    BILSTM_BERT_KNOWLEDGE = ""
    BERT = ""


    result_file = "../results/UKPSententialArgMin/wikidata_cypher/only_sub_and_inst_limit15/model_runs/bilstm_bert/TEST_result_twolabel_crossdomain_monitor-val_loss_do-0.2_lsize-128_bs-32_epochs-10_lr-0.001_attsize-50_trainemb-False_kl-only_sub_and_inst_limit15_kc-shallow_knowledge_we-bert768_kge-wiki_benjamin_100.json"
    np.set_printoptions(precision=2)
    NORMALIZED = False

    cm_dict, class_names = get_averaged_cm(result_file)


    plt.figure()
    if NORMALIZED == True:
        plot_confusion_matrices(cm_dict, "Baseline BiLSTM", class_names)
    else:
        plot_confusion_matrices(cm_dict, "Baseline BiLSTM", class_names)

    plt.show()