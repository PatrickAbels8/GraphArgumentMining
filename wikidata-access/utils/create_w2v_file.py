from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

"""
Create w2v file from entity2id and entity2vec files from openKE KGE framework
"""
def create_w2v_txt(entity2id_file, entity2vec_file, w2v_file, num_entities, vector_dim):
    with open(w2v_file, "a") as out:

        print("Load files")
        out.write(str(num_entities) + " " + str(vector_dim) + "\n")

        e2id = pd.read_csv(entity2id_file, delimiter="\t", skiprows=1, header=None)
        print("Shape of entity2id/relation2id: " + str(e2id.shape))
        e2v = np.loadtxt(entity2vec_file, delimiter="\t", usecols=range(vector_dim))
        print("Shape of entity2vec/relation2vec: " + str(e2v.shape))
        #pd.read_csv(model_settings['kg_embeddings'][0] + "entity2vec.vec", delimiter="\t",
        #            header=None, usecols=range(model_settings['kg_embeddings'][2])).values

        print("Create w2v file")
        for i in tqdm(range(len(e2id))):
            out.write(str(e2id.iloc[i, 0]) + " " + " ".join([str(val) for val in e2v[i]]))
            if i < len(e2id):
                out.write("\n")

def save_w2v_binary(w2v_txt_file, w2v_binary_file):
    if os.path.isfile(w2v_txt_file):
        word_vectors = KeyedVectors.load_word2vec_format(w2v_txt_file, binary=False)
        word_vectors.save_word2vec_format(w2v_binary_file, binary=True)
    else:
        print("Please create w2v txt file with create_w2v_txt() first!")


def glove_to_w2v_txt():
    with open("SensEmbed.txt", "w") as f_out, open("SensEmbed.bin", 'rb') as f_in:
        # dimension = -1
        # counter = 0
        f_out.write("3849894 400\n")
        for line in tqdm(f_in):
            # counter += 1
            splitLine = line.split()
            word = splitLine[0]
            embedding = " ".join([val.decode("utf-8") for val in splitLine[1:]])

            # if counter == 1:
            # dimension = len(splitLine[1:])

            f_out.write(word.decode("utf-8") + " " + str(embedding) + "\n")

        # print("dimension: " + str(dimension)+ "; counter: " + str(counter))

if __name__ == '__main__':
    path = "../embeddings/en/babelnet/"
    object = "SensEmbed" #entity

    #create_w2v_txt(path+object+"2id.txt", path+object+"2vec.vec",
     #              path+object+"_w2v.txt", 321, 50)
    save_w2v_binary(path+object+"_w2v.txt", path+object+"_w2v.bin")