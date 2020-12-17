# Leveraging knowledge

### API for retrieving and adding knowledge to a corpus, as well as training models
Requirements: Python 3.5 (or 3.6) and all packages in requirements.txt, as well as the spaCy model via _python -m spacy download en_core_web_sm_
       
    For python 3.6 do:
        sudo add-apt-repository ppa:jonathonf/python-3.6
        sudo apt-get update
        sudo apt-get install python3.6
        
        cd ~
        python3.6 -m venv virtualenv --without-pip
        cd virtualenv/
        source bin/activate
        curl https://bootstrap.pypa.io/get-pip.py | python3
        pip install -r requirements.txt
    

#### 1. Generate training data and train model: generate_data_and_train.py

Generates the training data with the knowledge retrieved. The knowledge-enriched data is stored at "results/[KB_NAME]/[CONFIG_NAME]".
Specify the kowledge config file to use in the model_configs at _configs/model_configs/[KB_NAME]/[CONFIG_NAME]_, as well as the model to use and other settings.

    python generate_data_and_train.py --config configs/model_configs/[CORPUS_NAME]/[KB_NAME]/[CONFIG_NAME] --predict-test 1

>>> BEFORE 10 epochs, 10 seeds, all topics in results/UKP/kg/wiki/
>>> WHILE py -3.6 generate_data_and_train.py --config configs/model_configs/UKPSententialArgMin/knowledge_graph/KBiLSTM_UKP_wikidata.json --predict-test 1
>>> AFTER safe model and processed in Desktop/seved/[epochs]_[seeds]_[topics] and save printed f1 in /stats.txt

knowledge_config in the config file can be set to:

    
"shallow_knowledge": Only use the first linked entity of a token
"shallow_paths": 
"full_paths": Attention over full paths

#### Preprocessing
- Get KGEs:  
        Wikidata: Take from fileserver (smb://fileserver.ukp.informatik.tu-darmstadt.de/data_repository/KnowledgeGraphEmbeddings) or train yourself with openKE (https://git.ukp.informatik.tu-darmstadt.de/argumentext-research/openke)
        WordNet: Take from fileserver (smb://fileserver.ukp.informatik.tu-darmstadt.de/data_repository/KnowledgeGraphEmbeddings) or train yourself with openKE (https://git.ukp.informatik.tu-darmstadt.de/argumentext-research/openke)
        NELL: WordNet: Take from fileserver (smb://fileserver.ukp.informatik.tu-darmstadt.de/data_repository/KnowledgeGraphEmbeddings) or train yourself with openKE (https://git.ukp.informatik.tu-darmstadt.de/argumentext-research/openke)  
        
- OPTIONAL: For Knowledge Graph Embeddings trained with OpenKE 
        
        To save memory, convert to binary w2v format with utils.create_w2v_file. The format will be recognized automatically while generating the training data.
   