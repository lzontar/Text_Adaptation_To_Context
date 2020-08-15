# -*- coding: utf-8 -*-
import statistics
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import spacy

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from nltk import tokenize
import syllables

def bag_of_words(sentence):
    return Counter(word.lower().strip('.,') for word in sentence.split(' '))


def split_into_sentences(text):
    return tokenize.sent_tokenize(text)


def sentence_length(sent):
    tokens = tokenize.word_tokenize(sent)
    return len(tokens)


def calc_sentence_similarity(text, desc=True, debug=False):
    sentences = split_into_sentences(text)

    c = CountVectorizer()

    bow_matrix = c.fit_transform(sentences)

    normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)

    similarity_graph = normalized_matrix * normalized_matrix.T
    similarity_graph.toarray()
    if debug:
        print(similarity_graph)

    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i], s) for i, s in enumerate(sentences)),
                  reverse=True), similarity_graph


def beautify_text(adapted_text, adaptation_dto, debug=False):
    text = adapted_text
    sentences = split_into_sentences(text)

    # Capitalize first word of each sentence
    text = ''.join(list(map(lambda s: s.capitalize(), sentences)))

    # TO-DO: what if there exists a lowercase occurrence in the text
    entities, entities_lower = adaptation_dto.entities()
    entity_pairs = zip(entities, entities_lower)

    for x in entity_pairs:
        if x[1] != x[0]:
            text = replace(x, text)

    return text


def replace(x, text):
    text = re.sub(r'\b' + x[1] + r'\b', x[0], text)
    text = re.sub(r'([^a-zA-Z\s]*)' + x[1] + r'([^a-zA-Z\s]*)', r'\1' + x[0] + r'\2', text)
    return text


def entities_from_text(text, debug=False):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    entities = []
    entities_lower = []
    for entity in doc.ents:
        entities.append(entity.text)
        entities_lower.append(entity.text.lower())

    return entities, entities_lower


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def mean_sentence_length(text):
    return statistics.mean(list(map(sentence_length, split_into_sentences(text))))


def extractive_summarization(text, top_n=5, debug=False):
    stop_words = stopwords.words('english')

    nlp = spacy.load('en_core_web_sm')

    doc = nlp(text)

    corpus = [sent.text.lower() for sent in doc.sents]

    cv = CountVectorizer(stop_words=list(stop_words))
    cv_fit = cv.fit_transform(corpus)
    word_list = cv.get_feature_names();
    count_list = cv_fit.toarray().sum(axis=0)

    """
    The zip(*iterables) function takes iterables as arguments and returns an iterator. 
    This iterator generates a series of tuples containing elements from each iterable. 
    Let's convert these tuples to {word:frequency} dictionary"""

    word_frequency = dict(zip(word_list, count_list))

    val = sorted(word_frequency.values())

    # Check words with higher frequencies
    higher_word_frequencies = [word for word, freq in word_frequency.items() if freq in val[-3:]]

    # gets relative frequencies of words
    higher_frequency = val[-1]
    for word in word_frequency.keys():
        word_frequency[word] = (word_frequency[word] / higher_frequency)

    # SENTENCE RANKING: the rank of sentences is based on the word frequencies
    sentence_rank = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequency.keys():
                if sent in sentence_rank.keys():
                    sentence_rank[sent] += word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent] = word_frequency[word.text.lower()]
            else:
                continue

    top_sentences = (sorted(sentence_rank.values())[::-1])
    top_sent = top_sentences[:top_n]

    # Mount summary
    summary = []
    for sent, strength in sentence_rank.items():
        if strength in top_sent:
            summary.append(sent)

    # return orinal text and summary
    return ". ".join(list(map(lambda x: x.text, summary)))
    # stop_words = stopwords.words('english')
    # summarize_text = []
    #
    # # Step 1 - Read text anc split it
    # sentences = tokenize.word_tokenize(text)
    #
    # # Step 2 - Generate Similary Martix across sentences
    # sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    #
    # # Step 3 - Rank sentences in similarity martix
    # sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    # scores = nx.pagerank(sentence_similarity_graph, max_iter=10000)
    #
    # # Step 4 - Sort the rank and pick top sentences
    # ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    # if debug:
    #     print("Indexes of top ranked_sentence order are ", ranked_sentence)
    #
    # for i in range(top_n):
    #     summarize_text.append(" ".join(ranked_sentence[i][1]))
    #
    # return ". ".join(summarize_text)

def flesch_reading_ease(text):
    n_word = len(tokenize.word_tokenize(text))
    n_sents = len(split_into_sentences(text))
    n_syl = 0
    for w in tokenize.word_tokenize(text):
        n_syl = n_syl + syllables.estimate(w)
    fre = 206.835 - 1.015 * (n_word / n_sents) - 84.6 * (n_syl / n_word)
    return fre