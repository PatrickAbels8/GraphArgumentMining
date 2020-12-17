import numpy as np
from scipy import spatial

label_setups = {
    "three_label": ['Argument_against', 'Argument_for', 'NoArgument'],
    "two_label": ['Argument', 'NoArgument'],
    "stance_only": ['Argument_against', 'Argument_for'],
    "ibm_two_label": ['1', '0']  # evidence, no evidence
}

def get_word_emb(word_index, embedding_lookup):
    try:
        return embedding_lookup[word_index]
    except KeyError:
        # print 'generate random word vector for ' + word
        return np.random.uniform(-0.01, 0.01, embedding_lookup.vector_size)

def get_avg_embedding(sentence, embedding_lookup, vocab_we):
    avg_emb = 0.0
    if len(sentence) == 0:
        return np.zeros(embedding_lookup.vector_size)
    else:
        for word in sentence:
            avg_emb = np.add(avg_emb, get_word_emb(vocab_we.get(word, vocab_we['UNK']), embedding_lookup))
        return np.divide(avg_emb, len(sentence))

def get_avg_embedding_for_topic_list(topic_list, embedding_lookup, vocab_we):
    # takes a list of topic strings, calculates the avg embedding of each list item, and returns the averaged topic embedding list
    prev_topic = ""
    prev_topic_vec = np.zeros(embedding_lookup.shape[1])
    final_list = []
    for topic in topic_list:
        if topic != prev_topic:
            prev_topic_vec = get_avg_embedding(topic.split('_'), embedding_lookup, vocab_we)
            prev_topic = topic
        final_list.append(prev_topic_vec)
    return np.array(final_list)


    avg_emb = 0.0
    if len(sentence) == 0:
        return np.zeros(embedding_lookup.vector_size)
    else:
        for word in sentence:
            avg_emb = np.add(avg_emb, get_word_emb(vocab_we.get(word, vocab_we['UNK']), embedding_lookup))
        return np.divide(avg_emb, len(sentence))

def get_path_lengths(paths):
    """
    :param paths: list of paths to topic or between entities
    :return: the lengths of all paths, empty list if there are no paths
    (path len=1 if there is only 1 concept uniting two entities)
    """
    path_lens = []
    for path in paths:
        temp = [concept for concept in path.split("->") if concept[0] == "[" and concept[-1] == "]"]
        path_lens.append(len(temp) - 2)
    return [-1] if len(path_lens) == 0 else path_lens

def get_label_index_from_setup(label, setup, dataset):
    """
    Takes as input the label of a sentence and returns the index, depending on the setup that is used
    :param label:
    :param setup:
    :return:
    """

    if dataset == "UKPSententialArgMin":
        if setup == "three_label":
            return label_setups["three_label"].index(label)
        elif setup == "two_label":
            new_label = "Argument" if label == 'Argument_against' or label == 'Argument_for' else 'NoArgument'
            return label_setups["two_label"].index(new_label)
        else:
            return None
    elif dataset == "IBMDebaterEvidenceSentences":
        # 0 = no evidence, 1 = evidence; needs to be switched, to fit UKP corpus labels
        return label_setups["ibm_two_label"].index(label)
    else:
        return None

def get_no_linked_token(paths):
    # takes first and last concept of all paths, removes duplicates, and returns the length
    temp = []
    for path in paths:
        temp.append(path.split("->")[0])
        temp.append(path.split("->")[-1])
    return len(list(set(temp)))

def get_cosine_sim_input(X, X_topic_string, embeddings_lookup, vocab_we):
    result = []
    for j in range(len(X)):
        sent_indices = X[j]
        topic_string = X_topic_string[j]
        sent_cosine = []
        for i in range(len(sent_indices)):
            if sent_indices[i] > 0:
                word_vec = embeddings_lookup[sent_indices[i]]
                sim = [1 - spatial.distance.cosine(get_avg_embedding(topic_string.split('_'), embeddings_lookup, vocab_we), word_vec)]
            else:
                sim = [0.0]
            sent_cosine.append(sim)
        result.append(np.array(sent_cosine))
    return np.array(result)

def get_avg_embedding_for_topics(topics_list, embeddings_lookup, vocab):
    result = np.zeros((len(topics_list), embeddings_lookup.shape[1]))
    for i, topic_string in enumerate(topics_list):
        result[i] = get_avg_embedding(topic_string.split('_'), embeddings_lookup, vocab)
    return result

def get_topic_token_cos(X, X_topic, embedding_lookup):
    result = []
    for j in range(len(X)):
        sent_indices = X[j]
        sent_cosine = []
        for i in range(len(sent_indices)):
            if sent_indices[i] > 0:
                sim = [1 - spatial.distance.cosine(embedding_lookup[X[j][i]], X_topic[j])]
            else:
                sim = [0.0]
            sent_cosine.append(sim)
        result.append(np.array(sent_cosine))
    return np.array(result)