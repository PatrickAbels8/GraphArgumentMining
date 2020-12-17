import os
import numpy as np
from utils.helper import make_dirs, invert_dictionary
import pickle
from tqdm import tqdm
from helpers.data_generation import helper

def generate_flattened_full_paths_data(PROCESSED_DIR, PROCESSED_FULL_PATHS_DIR, PROCESSED_FULL_PATHS_FLATTENED_DIR):
    """
    This method converts full path training data files to flattened ones, i.e. the concepts of all paths are taken (relations are removed)
    and added as a list of concepts. Thus one dimension will be removed and the shallow paths/knowledge models can be used on the data.
    NOTE: full_paths files have to be generated already (automatically done if a model is trained and the files cannot be found)
    """
    def prepad_list(concept_list, MAX_SENT_LEN):
        padding_len = MAX_SENT_LEN - len(concept_list)
        if padding_len > 0:
            concept_list = [0 for _ in range(padding_len)] + concept_list
        if padding_len < 0:
            concept_list = concept_list[-MAX_SENT_LEN:]
        return concept_list

    def get_vocab_for_index(index, path):
        with open(path + "vocab_kge.pkl", 'rb') as vocab_file:
            vocab_kge = pickle.load(vocab_file)
            vocab_kge = {y: x for x, y in vocab_kge.items()}
            print(vocab_kge.get(index, "Not found"))

    # create directory for resulting files
    make_dirs([PROCESSED_FULL_PATHS_FLATTENED_DIR])

    if len([file for file in os.listdir(PROCESSED_FULL_PATHS_FLATTENED_DIR) if
                  os.path.isfile(PROCESSED_FULL_PATHS_FLATTENED_DIR + file) and file.endswith("kX.npy")]) == 8:
        print("Flattened full_paths training data already generated -> SKIPPING")
    else:
        print("Generating flattened full_paths training data for all 8 topics.")

        # sanity check for instance_of and subclass_of
        #get_vocab_for_index(14437, PROCESSED_DIR)
        #get_vocab_for_index(8684, PROCESSED_DIR)

        # load full_path training data files
        gold_files = [file for file in os.listdir(PROCESSED_FULL_PATHS_DIR) if os.path.isfile(PROCESSED_FULL_PATHS_DIR+file)]

        # iterate over all topics
        for data_file in tqdm(gold_files):
            if os.path.isfile(PROCESSED_FULL_PATHS_FLATTENED_DIR + data_file):
                print(str(data_file) + " already exists -> skipping!")
                continue

            with open(PROCESSED_FULL_PATHS_DIR + data_file, "rb") as f:
                print("Loading: ", data_file)
                # load file
                training_data = np.load(f)
                print("Training data shape: ", training_data.shape)

                # calculate new flattened dimension (remove relations)
                try:
                    last_dim = training_data.shape[2]*(int((training_data.shape[3]+1)/2))
                except Exception as e:
                    print("Exception occured in generate_flattened_full_paths_data")
                    print(repr(e))
                    print(training_data)


                # create matrix that will hold the flattened data
                training_data_flat = np.zeros((training_data.shape[0], training_data.shape[1], last_dim), dtype=np.int32)

                #print(training_data.shape)
                #print(training_data_flat.shape)

                # iterate over the training data, flatten the paths, and add to the new matrix
                for sample_pos in range(training_data.shape[0]):
                    for token_pos in range(training_data.shape[1]):
                        concept_list = []
                        for path_pos in range(training_data.shape[2]):
                            for concept_pos in range(training_data.shape[3]):
                                if concept_pos % 2 == 0:
                                    concept = training_data[sample_pos][token_pos][path_pos][concept_pos]
                                    if concept != 0:
                                        concept_list.append(concept)
                        #assert 14437 not in concept_list and 8684 not in concept_list, "There are still relations in concept_list => error detected"
                        concept_list = prepad_list(concept_list, last_dim) # pre pad with zeros
                        training_data_flat[sample_pos][token_pos] = np.array(concept_list)

                # save new matrix
                np.save(PROCESSED_FULL_PATHS_FLATTENED_DIR+data_file, training_data_flat)

def generate_BERT_training_data(vocab_we, setting, SOURCE_DIR, TARGET_DIR, BERT_SERVER_IP):
    """
    This class converts indexed training data into training data that contains BERT embeddings, thus models used with this data have to have
    their embedding layer removed.
    :param vocab_we: The vocab used for the indexed data
    :param SOURCE_DIR: The directory which contains the processed sentences
    :param TARGET_DIR: Save the converted sentences here
    """

    # create directory for resulting files
    make_dirs([TARGET_DIR])

    # check if files exist
    if len([file for file in os.listdir(TARGET_DIR) if
                  os.path.isfile(TARGET_DIR + file) and file.endswith("_X.npy")]) == 8:
        print("BERT training data already created -> SKIPPING")
    else:
        print("Generating BERT training data for all 8 topics.")
        print("Please make sure to start the BERT service with POOLING_STRATEGY=NONE for BERT word embeddings!")

        # invert vocab_we
        inverted_vocab_we = invert_dictionary(vocab_we)

        # load full_path training data files
        gold_files = [file for file in os.listdir(SOURCE_DIR) if
                      os.path.isfile(SOURCE_DIR + file) and (file.endswith("_X.npy") or file.endswith("_kX.npy"))]

        # iterate over all topics
        for data_file in tqdm(gold_files):
            if os.path.isfile(TARGET_DIR + data_file):
                print(str(data_file) + " already exists -> skipping!")
                continue

            with open(SOURCE_DIR + data_file, "rb") as f:
                # load file
                training_data = np.load(f)

                # convert to BERT
                training_data_bert = helper.convert_X_to_BERT(training_data, inverted_vocab_we, setting, BERT_SERVER_IP)

                # save file into new location
                np.save(TARGET_DIR+data_file, training_data_bert)