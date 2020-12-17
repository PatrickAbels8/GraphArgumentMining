import json
import nltk
import os
import copy
from helpers.classification import generate_features


def add_metadata_from_split(split_dict, stats_dict):
    for topic, value in split_dict.items():
        stats_dict[topic]['metadata'] = {'train_len': len(value['train_indices']),
                                         'dev_len': len(value['dev_indices']),
                                         'test_len': len(value['test_indices'])}


def init_stats_dict(SAMPLES_DIR, format="csv", dataset="UKPSententialArgMin"):
    # add general stats about knowledge enriched data
    gold_files = [file for file in os.listdir(SAMPLES_DIR) if os.path.isfile(SAMPLES_DIR + file) and file.startswith("result_")]
    stats_dict = {}
    for data_file in gold_files:
        topic_name = data_file.split("result_")[1].split("_nx")[0]

        if dataset == "UKPSententialArgMin":
            stats_dict[topic_name] = {'All': {}, 'Argument_for': {}, 'Argument_against': {}, 'NoArgument': {}}
        elif dataset == "IBMDebaterEvidenceSentences":
            stats_dict[topic_name] = {'All': {}, '0': {}, '1': {}}
        else:
            raise Exception("Dataset not recognized!")

        if format == "csv":
            for key in stats_dict[topic_name].keys(): #
                stats_dict[topic_name][key] = {
                    "abs": {},
                    "rel": {}
                }
                stats_dict[topic_name][key]['abs']['#tokens'] = 0
                stats_dict[topic_name][key]['abs']['#sents'] = 0
                stats_dict[topic_name][key]['abs']['#tokens_linked_within_sent'] = 0
                stats_dict[topic_name][key]['abs']['#sents_with_paths_between_entities'] = 0
                stats_dict[topic_name][key]['abs']['#tokens_linked_to_topic'] = 0
                stats_dict[topic_name][key]['abs']['#sents_with_paths_to_topic'] = 0
                stats_dict[topic_name][key]['abs']['#paths_between_entities'] = 0
                stats_dict[topic_name][key]['abs']['#paths_to_topic'] = 0
        elif format == "tex":
            for key in stats_dict[topic_name].keys(): #
                stats_dict[topic_name][key]['#sents'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#tokens'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#tokens_linked_within_sent'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#tokens_linked_to_topic'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#sents_with_paths_between_entities'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#sents_with_paths_to_topic'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#paths_between_entities'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#paths_to_topic'] = {"abs": 0, "rel": 0}
                stats_dict[topic_name][key]['#include_topic'] = {"abs": 0, "rel": 0} # +1 if there is at least one path with l=1 between a topic and sentence concept
                stats_dict[topic_name][key]['#include_only_topic'] = {"abs": 0, "rel": 0} # +1 if there are only paths with l=1 between a topic and sentence concept
        else:
            print('Please specify format as "csv" or "tex"')

    stats_dict['metadata'] = {}
    return stats_dict

def generate_tex_knowledge_stats(SAMPLES_DIR, stats_dict):
    header = r'''
    \documentclass{article}
    \usepackage{graphicx}
    \usepackage{hyperref}
    \usepackage{tabularx}
    \usepackage{pdflscape}
    \usepackage{afterpage}
    \usepackage{booktabs}
    \usepackage{multirow}
    \newcommand\model[1]{\textsf{#1}}
    \usepackage{makecell}
    \usepackage[top=0in, bottom=0in, left=0in, right=0in]{geometry}
    \begin{document}

    '''




    def get_formatted_stat_for_label(info):
        # format stats for one topic and one label (all, pro, con, no_arg) to csv
        out = ""
        for i, value in enumerate(info.values()):
            out += str(value) + (";" if i < len(info.values()) else "")
        return out

    def format_table_column(column_name, alignment=None, makecell=True):
        if makecell == True:
            if column_name.count(r'_') > 1:
                split_string = column_name.rsplit(r'_', 2)
                column_name = get_tex_make_cell(split_string[0], split_string[1] + " " + split_string[2], alignment=alignment)
        return column_name.replace("_", r' ').replace("#", r'\#')

    def get_tex_make_cell(first, second, alignment=None):
        if alignment not in ["c", "l", "r"]:
            return r'\makecell{' + str(first) + r'\\' + str(second) + r'}'
        else:
            return r'\makecell['+str(alignment)+r']{' + str(first) + r'\\' + str(second) + r'}'

    def print_main_table(stats_dict, column_length, exclude_list, table_lbl):
        # print version

        footer = r'''
		\end{tabular}
		\caption{Numbers in brackets below values are relative values (per sentence).}
		\label{tab:'''+table_lbl+'''}
        \end{table*} 
        '''
        table_begin = r'''
        
        \begin{table*}
        
        \centering
        \setlength{\tabcolsep}{0.25em}
        \def\arraystretch{1.3}
        \footnotesize
        '''

        # create copy of dict
        stats_dict_temp = copy.deepcopy(stats_dict)

        # remove unwanted columns
        for topic, values in stats_dict_temp.items():
            if topic != "metadata":
                for label, cols in values.items():
                    if label != "metadata":
                        for c_name in list(cols.keys()): # otherwise dict changes size while iteration
                            if c_name in exclude_list:
                                cols.pop(c_name, None)

        table_begin += r'\begin{tabular}{ @{}lc' + ''.join(["c"] * column_length) + r'@{}}' + '\n' + r'\toprule' + '\n'
        # table_begin += r'&&\multicolumn{'+str(NUM_ABS)+r'}{c}{Absolute}& \multicolumn{'+str(NUM_REL)+r'}{c}{Average (per sent)} \\'+'\n'
        # table_begin += r'\cmidrule(r){3-'+str(3+NUM_ABS-1)+r'} ' +'\n' \
        #               r'\cmidrule(l){'+str(3+NUM_ABS)+'-'+str(3+NUM_ABS+NUM_REL-1)+'}' +'\n'

        table_begin += r'topic&label&' + \
                       r'&'.join([format_table_column(column_name) for column_name in
                                  stats_dict_temp[list(stats_dict_temp.keys())[0]]['All'].keys()]) + \
                       r'\\' + '\n' + r'\midrule' + '\n'
        body = ""
        for j, (topic_name, label_info) in enumerate(stats_dict_temp.items()):
            if topic_name != 'metadata':
                body += r'\multirow{4}{*}{'+format_table_column(topic_name)+r'}'
                for key, columns in label_info.items():
                    if key != 'metadata':
                        body += " & " + format_table_column(key) + " & "

                        for i, (col_name, col_values) in enumerate(columns.items()):
                            if col_name == "#sents":
                                body += get_tex_make_cell(col_values['abs'], "")
                            elif col_name in ["#sents_with_paths_between_entities", "#sents_with_paths_to_topic", "#include_topic", "#include_only_topic"]:
                                body += get_tex_make_cell("{0:.2f}".format(col_values['abs']), "(" + str(int(col_values['rel']*100)) + r"\%)")
                            else:
                                body += get_tex_make_cell("{0:.2f}".format(col_values['abs']), "("+ str("{0:.2f}".format(col_values['rel'])) + ")")
                            body += (" & " if i < len(columns.items())-1 else r'\\')
                body += '\n' + (r'\bottomrule' if j == len(stats_dict_temp.items())-1 else r'\midrule') + '\n'
                    #body += (";" if i > 0 else "") + key + ";" + get_formatted_stat_for_label(label_info[key]) + "\n"


        return table_begin + body + footer

    def print_meta_data_table(stats_dict):
        table = r'''
        \begin{table*}
	
        \centering
        \setlength{\tabcolsep}{0.25em}
        \def\arraystretch{1.3}
        \footnotesize

		\begin{tabular}{ @{}lc@{}}
        \toprule
        name& value  \\
        \midrule
        '''
        for i, (key, value) in enumerate(stats_dict['metadata'].items()):
            table += format_table_column(key, makecell=False) + " & " + str(value) + ("\%" if "rel" in key else "") + r'\\' + '\n' + r'\midrule' + '\n'

        for i, (topic_name, values) in enumerate(stats_dict.items()):
            if topic_name != 'metadata':
                table += format_table_column(topic_name+'_train_len', makecell=False) + " & " + str(values['metadata']['train_len']) + r'\\' + '\n'
                table += format_table_column(topic_name+'_dev_len', makecell=False) + " & " + str(values['metadata']['dev_len']) + r'\\' + '\n'
                table += format_table_column(topic_name+'_test_len', makecell=False) + " & " + str(values['metadata']['test_len']) + r'\\' + '\n'
                table += (r' \bottomrule ' if i == len(stats_dict.items())-1 else r' \midrule ')

        table += r'''
        \end{tabular}	
		\caption{Some metdadata of the training data preprocessing.}
		\label{tab:metadata_results}
        \end{table*} 
        '''

        return table

    print("Creating statistics for data at " + SAMPLES_DIR)

    # add general stats about knowledge enriched data
    gold_files = [file for file in os.listdir(SAMPLES_DIR) if os.path.isfile(SAMPLES_DIR + file) and file.startswith("result_")]

    for data_file in gold_files:
        topic_name = data_file.split("result_")[1].split("_nx")[0]

        with open(SAMPLES_DIR + data_file, "r") as f:
            # load data and get number of samples for the topic
            topic_data = json.load(f)

            for sample in topic_data['samples']:# get absolute values
                if sample['label'] != 'metadata':

                    stats_dict[topic_name][sample['label']]['#sents']['abs'] += 1
                    stats_dict[topic_name]['All']['#sents']['abs'] += 1

                    stats_dict[topic_name][sample['label']]['#tokens']['abs'] += len(nltk.word_tokenize(sample['sentence']))
                    stats_dict[topic_name]['All']['#tokens']['abs'] += len(nltk.word_tokenize(sample['sentence']))  # TODO remove [0] when fixed at knowledge retrieval

                    stats_dict[topic_name][sample['label']]['#paths_between_entities']['abs'] += len(sample['paths_between_entities'])
                    stats_dict[topic_name]['All']['#paths_between_entities']['abs'] += len(sample['paths_between_entities'])

                    stats_dict[topic_name][sample['label']]['#paths_to_topic']['abs'] += len(sample['paths_to_topic'])
                    stats_dict[topic_name]['All']['#paths_to_topic']['abs'] += len(sample['paths_to_topic'])

                    stats_dict[topic_name][sample['label']]['#sents_with_paths_between_entities']['abs'] += (1 if len(sample['paths_between_entities']) > 0 else 0)
                    stats_dict[topic_name]['All']['#sents_with_paths_between_entities']['abs'] += (1 if len(sample['paths_between_entities']) > 0 else 0)

                    stats_dict[topic_name][sample['label']]['#sents_with_paths_to_topic']['abs'] += (1 if len(sample['paths_to_topic']) > 0 else 0)
                    stats_dict[topic_name]['All']['#sents_with_paths_to_topic']['abs'] += (1 if len(sample['paths_to_topic']) > 0 else 0)

                    stats_dict[topic_name][sample['label']]['#tokens_linked_within_sent']['abs'] += generate_features.get_no_linked_token(sample['paths_between_entities'])
                    stats_dict[topic_name]['All']['#tokens_linked_within_sent']['abs'] += generate_features.get_no_linked_token(sample['paths_between_entities'])

                    stats_dict[topic_name][sample['label']]['#tokens_linked_to_topic']['abs'] += generate_features.get_no_linked_token(sample['paths_to_topic'])
                    stats_dict[topic_name]['All']['#tokens_linked_to_topic']['abs'] += generate_features.get_no_linked_token(sample['paths_to_topic'])

                    # TODO
                    stats_dict[topic_name][sample['label']]['#include_only_topic']['abs'] += (1 if max(
                        generate_features.get_path_lengths(sample['paths_to_topic'])) == 1 else 0)
                    stats_dict[topic_name]['All']['#include_only_topic']['abs'] += (1 if max(
                        generate_features.get_path_lengths(sample['paths_to_topic'])) == 1 else 0)

                    stats_dict[topic_name][sample['label']]['#include_topic']['abs'] += (1 if 1 in generate_features.get_path_lengths(sample['paths_to_topic']) else 0)
                    stats_dict[topic_name]['All']['#include_topic']['abs'] += (1 if 1 in generate_features.get_path_lengths(sample['paths_to_topic']) else 0)

            # get some relative values
            for key in stats_dict[topic_name].keys():
                if key != 'metadata':
                    total_samples = stats_dict[topic_name][key]['#sents']['abs']
                    if total_samples == 0:
                        total_samples = 1 # prevent division by zero
                    stats_dict[topic_name][key]['#tokens']['rel'] = stats_dict[topic_name][key]['#tokens']['abs'] / total_samples
                    stats_dict[topic_name][key]['#tokens_linked_within_sent']['rel'] = stats_dict[topic_name][key]['#tokens_linked_within_sent']['abs'] / total_samples
                    stats_dict[topic_name][key]['#sents_with_paths_between_entities']['rel'] = stats_dict[topic_name][key]['#sents_with_paths_between_entities']['abs'] / total_samples
                    stats_dict[topic_name][key]['#sents_with_paths_to_topic']['rel'] = stats_dict[topic_name][key]['#sents_with_paths_to_topic']['abs'] / total_samples
                    stats_dict[topic_name][key]['#tokens_linked_to_topic']['rel'] = stats_dict[topic_name][key]['#tokens_linked_to_topic']['abs'] / total_samples
                    stats_dict[topic_name][key]['#tokens_linked_within_sent']['rel'] = stats_dict[topic_name][key]['#tokens_linked_within_sent']['abs'] / total_samples
                    stats_dict[topic_name][key]['#paths_between_entities']['rel'] = stats_dict[topic_name][key]['#paths_between_entities']['abs'] / total_samples
                    stats_dict[topic_name][key]['#paths_to_topic']['rel'] = stats_dict[topic_name][key]['#paths_to_topic']['abs'] / total_samples
                    stats_dict[topic_name][key]['#include_only_topic']['rel'] = stats_dict[topic_name][key]['#include_only_topic']['abs'] / total_samples
                    stats_dict[topic_name][key]['#include_topic']['rel'] = stats_dict[topic_name][key]['#include_topic']['abs'] / total_samples

    result = header

    exclude_list = ["#include_topic", "#include_only_topic"]
    result += print_main_table(stats_dict, len(stats_dict[list(stats_dict.keys())[0]]['All'])-len(exclude_list), exclude_list, "results_full")

    include_list = ["#sents_with_paths_to_topic", "#include_topic", "#include_only_topic"]
    all_column_names = list(stats_dict[list(stats_dict.keys())[0]]['All'].keys())
    exclude_list = [col for col in all_column_names if col not in include_list]
    result += print_main_table(stats_dict, len(include_list), exclude_list, "results_topic")
    if len(stats_dict['metadata'].items()) > 0:
        result += print_meta_data_table(stats_dict)
        result += r'\end{document}'
    else:
        result += r'\end{document}'
    return result, stats_dict

def generate_csv_knowledge_stats(SAMPLES_DIR, PROCESSED_DIR, stats_dict):
    def get_no_linked_token(paths):
        # takes first and last concept of a path, removes duplicates, and returns the length
        temp = []
        for path in paths:
            temp.append(path.split("->")[0])
            temp.append(path.split("->")[-1])
        return len(list(set(temp)))

    def get_formatted_stat_for_label(info):
        # format stats for one topic and one label (all, pro, con, no_arg) to csv
        out = ""
        for i, value in enumerate(info.values()):
            out += str(value) + (";" if i < len(info.values()) else "")
        return out

    def format_table_column(column_name):
        if column_name.count(r'_') > 1:
            split_string = column_name.rsplit(r'_', 2)
            column_name = r'\makecell{' + split_string[0] + r'\\' + split_string[1] + " " + split_string[2] + r'}'
        return column_name.replace("_", r' ').replace("#", r'\#')

    def print_csv(stats_dict, SAMPLES_DIR, PROCESSED_DIR):
        # print version
        header = "Stats for data at " + SAMPLES_DIR + "\n"
        header += "topic;label;" + ";".join(
            [column_name for column_name in stats_dict[list(stats_dict.keys())[0]]['All']['abs']])
        header += "topic;label;" + ";".join(
            [column_name for column_name in stats_dict[list(stats_dict.keys())[0]]['All']['rel']]) + "\n"
        body = ""
        # TODO add relative stuff like paths/sent, etc.
        for topic_name, info in stats_dict.items():
            if topic_name != 'metadata':
                body += topic_name + ";"
                for i, key in enumerate(info.keys()):
                    body += (";" if i > 0 else "") + key + ";" + \
                            get_formatted_stat_for_label(info[key]['abs']) + ";" + \
                            get_formatted_stat_for_label(info[key]['rel']) + "\n"

        # TODO add metadata: number of OOV for KGE, WE, etc...

        print("Write file with statistics to " + PROCESSED_DIR + SAMPLES_DIR.split("/")[1] + '_stats.csv')
        return header + body

    print("Creating statistics for data at " + SAMPLES_DIR)

    # add general stats about knowledge enriched data
    gold_files = [file for file in os.listdir(SAMPLES_DIR) if os.path.isfile(SAMPLES_DIR + file)]

    for data_file in gold_files:
        topic_name = data_file.split("result_")[1].split("_nx")[0]

        with open(SAMPLES_DIR + data_file, "r") as f:
            # load data and get number of samples for the topic
            topic_data = json.load(f)

            for sample in topic_data['samples']:  # get absolute values
                stats_dict[topic_name][sample['label']]['abs']['#sents'] += 1
                stats_dict[topic_name]['All']['abs']['#sents'] += 1

                stats_dict[topic_name][sample['label']]['abs']['#tokens'] += len(
                    nltk.word_tokenize(sample['sentence']))
                stats_dict[topic_name]['All']['abs']['#tokens'] += len(
                    nltk.word_tokenize(sample['sentence']))  # TODO remove [0] when fixed at knowledge retrieval

                stats_dict[topic_name][sample['label']]['abs']['#paths_between_entities'] += len(
                    sample['paths_between_entities'])
                stats_dict[topic_name]['All']['abs']['#paths_between_entities'] += len(sample['paths_between_entities'])

                stats_dict[topic_name][sample['label']]['abs']['#paths_to_topic'] += len(sample['paths_to_topic'])
                stats_dict[topic_name]['All']['abs']['#paths_to_topic'] += len(sample['paths_to_topic'])

                stats_dict[topic_name][sample['label']]['abs']['#sents_with_paths_between_entities'] += (
                1 if len(sample['paths_between_entities']) > 0 else 0)
                stats_dict[topic_name]['All']['abs']['#sents_with_paths_between_entities'] += (
                1 if len(sample['paths_between_entities']) > 0 else 0)

                stats_dict[topic_name][sample['label']]['abs']['#sents_with_paths_to_topic'] += (
                1 if len(sample['paths_to_topic']) > 0 else 0)
                stats_dict[topic_name]['All']['abs']['#sents_with_paths_to_topic'] += (
                1 if len(sample['paths_to_topic']) > 0 else 0)

                stats_dict[topic_name][sample['label']]['abs']['#tokens_linked_within_sent'] += get_no_linked_token(
                    sample['paths_between_entities'])
                stats_dict[topic_name]['All']['abs']['#tokens_linked_within_sent'] += get_no_linked_token(
                    sample['paths_between_entities'])

                stats_dict[topic_name][sample['label']]['abs']['#tokens_linked_to_topic'] += get_no_linked_token(
                    sample['paths_to_topic'])
                stats_dict[topic_name]['All']['abs']['#tokens_linked_to_topic'] += get_no_linked_token(
                    sample['paths_to_topic'])

            # get some relative values
            for key in stats_dict[topic_name].keys():
                stats_dict[topic_name][key]['rel']['tokens_per_sent'] = stats_dict[topic_name][key]['abs']['#tokens'] / \
                                                                        stats_dict[topic_name][key]['abs']['#sents']
                stats_dict[topic_name][key]['rel']['avg(tokens_linked_within_sent)'] = \
                stats_dict[topic_name][key]['abs']['#tokens_linked_within_sent'] / stats_dict[topic_name][key]['abs'][
                    '#sents']
                stats_dict[topic_name][key]['rel']['avg(sents_with_paths_between_entities)'] = \
                stats_dict[topic_name][key]['abs']['#sents_with_paths_between_entities'] / \
                stats_dict[topic_name][key]['abs']['#sents']
                stats_dict[topic_name][key]['rel']['avg(sents_with_paths_to_topic)'] = \
                stats_dict[topic_name][key]['abs']['#sents_with_paths_to_topic'] / stats_dict[topic_name][key]['abs'][
                    '#sents']
                stats_dict[topic_name][key]['rel']['avg(tokens_linked_to_topic)'] = stats_dict[topic_name][key]['abs'][
                                                                                        '#tokens_linked_to_topic'] / \
                                                                                    stats_dict[topic_name][key]['abs'][
                                                                                        '#sents']
                stats_dict[topic_name][key]['rel']['avg(tokens_linked_within_sent)'] = \
                stats_dict[topic_name][key]['abs']['#tokens_linked_within_sent'] / stats_dict[topic_name][key]['abs'][
                    '#sents']
                stats_dict[topic_name][key]['rel']['avg(paths_between_entities)'] = stats_dict[topic_name][key]['abs'][
                                                                                        '#paths_between_entities'] / \
                                                                                    stats_dict[topic_name][key]['abs'][
                                                                                        '#sents']
                stats_dict[topic_name][key]['rel']['avg(paths_to_topic)'] = stats_dict[topic_name][key]['abs'][
                                                                                '#paths_to_topic'] / \
                                                                            stats_dict[topic_name][key]['abs']['#sents']


    return print_csv(stats_dict, SAMPLES_DIR, PROCESSED_DIR)

def create_and_print_knowledge_stats(SAMPLES_DIR, PROCESSED_DIR, stats_dict, format="csv"):
    if format == "csv":
        printout = generate_csv_knowledge_stats(SAMPLES_DIR, PROCESSED_DIR, stats_dict)
        with open(PROCESSED_DIR + SAMPLES_DIR.split("/")[1] + '_stats.csv', 'a') as f:
            f.write(printout)
    elif format == "tex":
        printout, stats_dict = generate_tex_knowledge_stats(SAMPLES_DIR, stats_dict)
        print("Write file with statistics to " + PROCESSED_DIR + SAMPLES_DIR.split("/")[1] + '_stats.tex')
        with open(PROCESSED_DIR + SAMPLES_DIR.split("/")[1] + '_stats.tex', 'w') as f:
            f.write(printout)
        return stats_dict
    else:
        print('No printout created! Please specify format as "csv" or "tex"')

