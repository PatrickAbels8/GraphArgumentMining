import os
from pytz import timezone
import urllib3
from datetime import datetime
import pickle
import errno
import requests
from urllib.error import HTTPError
import time
import urllib.parse, urllib.request
import json
from json import JSONDecodeError

def invert_dictionary(dict):
    return {y:x for x, y in dict.items()}

def make_dirs(dirs_list):
    for directory in dirs_list:
        try:
            os.makedirs(directory)
        except IOError as e:
            print(e)

def get_time():
    cet = timezone('Europe/Berlin')
    cet_time = datetime.now(cet)
    return cet_time.strftime("%Y-%m-%d %H:%M:%S")

def save_as_pickle(data, path):
    try:
        os.makedirs(path.rsplit("/",1)[0])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(path):
    with open(path, "rb") as input_file:
        file = pickle.load(input_file)
    return file

def handle_request(query):
    query_counter = 0
    max_retries = 20
    delay = 10

    while query_counter < max_retries:
        try:
            response = requests.get(query, timeout=60).json()
            break
        except HTTPError as he:
            query_counter += 1
            print("HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)
        except JSONDecodeError as jd:
            query_counter += 1
            print("JSONDecodeError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)

    if query_counter >= max_retries:
        print("Max retries reached, exit!")
        exit(1)

    return response

def handle_request_babelnet(query, url, method="POST"):
    query_counter = 0
    max_retries = 20
    delay = 10

    while query_counter < max_retries:
        try:
            response = requests.get(url, params=query.encode("utf8"), timeout=60).json()
            break
        except HTTPError as he:
            query_counter += 1
            print("HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            print(str(he))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)
        except JSONDecodeError as jd:
            query_counter += 1
            print("JSONDecodeError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)

    if query_counter >= max_retries:
        print("Max retries reached, exit!")
        exit(1)

    return response

def handle_urllib_request(query, url, method="POST"):
    query_counter = 0
    max_retries = 20
    delay = 10

    req = urllib.request.Request(url, data=query.encode("utf8"), method=method)
    while query_counter < max_retries:
        try:
            with urllib.request.urlopen(req, timeout = 60) as f:
                response = f.read()
                response = json.loads(response.decode("utf8"))
                query_counter = max_retries
        except HTTPError as he:
            query_counter += 1
            print("HTTPError while querying for entities. Try again: " + str(query_counter) + "/" + str(max_retries))
            if query_counter == max_retries:
                exit(1)
            else:
                time.sleep(delay)
    return response
