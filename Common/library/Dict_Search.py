import requests
import re
import sys
import nltk
from PyDictionary import PyDictionary
from nltk.corpus import stopwords


def get_definition(word):
    dictionary = PyDictionary()

    word = word.lower()

    try:

        word_definition = dictionary.meaning(word)[list(dictionary.meaning(word).keys())[0]][0]
        if (word_definition.find('],') != -1):
            word_definition = word_definition.split('],')[0]

        bc_index = word_definition.find('{bc}')

        end_index = word_definition.find('}:')
        end_index_1 = word_definition.find('\']]')
        if (end_index_1 < end_index and end_index_1 > 0):
            end_index = end_index_1

        end_index_2 = word_definition.find('\'],')
        if (end_index_2 < end_index):
            end_index = end_index_2

        if (end_index == -1):
            end_index = len(word_definition)

        word_definition = word_definition[bc_index + 4: end_index]

        word_definition = re.sub('\|.*?\|', '|', word_definition)
        word_definition = re.sub('{.*?\|', '', word_definition)
        word_definition = re.sub('{.*?}', '', word_definition)
        word_definition = re.sub('[^a-zA-Z ]+', '', word_definition)
        word_definition = re.sub(' +', ' ', word_definition)

        return word_definition
    except:
        return ''


def get_n_of_words(word):
    stop_words = set(stopwords.words('english'))

    word_depth_value = 0

    known_words = set()
    known_words.add(word)

    unknown_words = set()
    word_definition = get_definition(word)

    word_definition_arr = word_definition.split(" ")
    for word in word_definition_arr:
        if word not in stop_words and len(word) > 1:
            unknown_words.add(word)

    while len(unknown_words) > 0:
        word = unknown_words.pop()
        known_words.add(word)
        word_definition = get_definition(word)
        word_definition_arr = word_definition.split(" ")

        for word in word_definition_arr:
            if word not in known_words and word not in unknown_words and word not in stop_words:
                unknown_words.add(word)
                word_depth_value += 1

    return word_depth_value
    print("I needed to learn " + str(word_depth_value) + " words")

