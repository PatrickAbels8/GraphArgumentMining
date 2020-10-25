import os
from pathlib import Path
from collections import defaultdict

module_location = Path(os.path.dirname(__file__))
RESOURCES_FOLDER = str(module_location)+ "/"+ "resources/wikidata_ukp/"


def load_list(path_to_list):
    """
    Load a set of labels from a file. Each line corresponds to a single label.

    :param path_to_list: path to the file
    :return: the set of the labels
    """
    with open(path_to_list, encoding="utf-8") as f:
        return_list = {l.strip() for l in f.readlines()}
    return return_list


def load_property_labels(path_to_property_labels):
    """
    Load descriptions for wikidata relations. The output is a dictionary mapping from a relation id to teh description.

    :param path_to_property_labels: path to the file
    :return: a dictionary from relation ids to descriptions
    >>> load_property_labels(RESOURCES_FOLDER + "properties_with_labels.txt")["P106"]
    {'type': 'wikibase-item', 'altlabel': ['employment', 'craft', 'profession', 'job', 'work', 'career'], 'freq': 2290043, 'label': 'occupation'}
    >>> load_property_labels(RESOURCES_FOLDER + "properties_with_labels.txt")["P1014564"]
    {'freq': 0}
    """
    with open(path_to_property_labels, encoding="utf-8") as infile:
        return_map = defaultdict(lambda: {'freq': 0, 'type': None})
        for l in infile.readlines():
            if not l.startswith("#"):
                columns = l.split("\t")
                return_map[columns[0].strip()] = {"label": columns[1].strip().lower(),
                                                  "altlabel": list(set(columns[3].strip().lower().split(", "))),
                                                  "type":  columns[4].strip().lower(),
                                                  "freq": int(columns[5].strip().replace(",",""))}
    return return_map


WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
WIKIDATA_PROPERTY_PREFIX = "http://www.wikidata.org/prop/direct/"
WIKIPEDIA_PREFIX = "https://en.wikipedia.org/wiki/"
property_blacklist = load_list(RESOURCES_FOLDER + "property_blacklist.txt")
property2label = load_property_labels(RESOURCES_FOLDER + "properties_with_labels.txt")
BLACKSET_PROPERTY_OBJECT_TYPES = {'commonsmedia',
                                  'external-id',
                                  'globe-coordinate',
                                  'math',
                                  'monolingualtext',
                                  'quantity',
                                  'string',
                                  'url',
                                  'wikibase-property'}

content_properties = {p for p, v in property2label.items() if v.get("type") not in BLACKSET_PROPERTY_OBJECT_TYPES
                       and v.get('freq') > 5
                       and 'category' not in v.get('label', "")
                       and 'taxon' not in v.get('label', "")} - property_blacklist

frequent_properties = {p for p in content_properties if property2label[p].get('freq') > 1000}
