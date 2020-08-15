import statistics

import torch
import requests
import xml.etree.ElementTree as ET
from rake_nltk import Metric, Rake
from nltk.corpus import stopwords
from Common.classes.TextRank4Keyword import TextRank4Keyword
import Common.library.Common as com
from nltk import tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
from Common.classes.RNN_Model import Prediction, Preprocessing


# https://huggingface.co/transformers/model_doc/bart.html?highlight=no_repeat_ngram_size


def evaluate_summary(origin_text):
    pass


def extractive_summarization(adaptation_dto, mean_measures, epsilon, debug):
    text = adaptation_dto.adapted_text()
    text_measures = adaptation_dto.text_measures()
    mean_sent_length = com.mean_sentence_length(text)

    abs_diff_length = abs(mean_measures['LEN'][adaptation_dto.target_pub_type()] - text_measures['LEN'])

    summary = com.extractive_summarization(adaptation_dto.adapted_text(),
                                           top_n=math.floor(abs_diff_length / mean_sent_length), debug=debug)
    adaptation_dto.adapted_text(summary)
    return adaptation_dto


def summarize(adaptation_dto, mean_measures, epsilon, debug=False):
    text = adaptation_dto.adapted_text()

    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cpu')

    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_text = "summarize: " + preprocess_text

    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)

    a_tokens = math.floor(-epsilon * mean_measures["LEN"][adaptation_dto.target_pub_type()] \
                          + mean_measures["LEN"][adaptation_dto.target_pub_type()])
    b_tokens = math.ceil(epsilon * mean_measures["LEN"][adaptation_dto.target_pub_type()] \
                         + mean_measures["LEN"][adaptation_dto.target_pub_type()])
    if debug:
        print("Text measures before summarization: ", adaptation_dto.text_measures())

    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=b_tokens if a_tokens > b_tokens else a_tokens,
                                 max_length=a_tokens if a_tokens > b_tokens else b_tokens,
                                 early_stopping=False,
                                 repetition_penalty=2.5,
                                 length_penalty=1.50)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=False)

    if debug:
        print("\n\n    ------------------------------- ORIGINAL TEXT -------------------------------    \n\n",
              adaptation_dto.adapted_text())

    if debug:
        print("\n\n    ------------------------------- SUMMARY -------------------------------    \n\n", summary)

    adaptation_dto.adapted_text(summary)

    if debug:
        print("Text measures after summarization: ", adaptation_dto.text_measures())

    return summary


def similar_sentences(orig_sents, extra_sents, debug):
    similar_sentences = []
    for x in orig_sents:
        for y in extra_sents:
            similar_sentences.append((x, y, com.sentence_similarity(x, y)))

    similar_sentences.sort(key=lambda x: x[2], reverse=True)
    return similar_sentences


def add_similar_sentences(adaptation_dto, mean_measures, extra_content, rel_sent_length, debug):
    text = adaptation_dto.adapted_text()
    text_measures = adaptation_dto.text_measures()
    list_of_lists_of_sents = list(map(lambda x: tokenize.sent_tokenize(x), extra_content))
    curr_diff = (mean_measures['LEN'][adaptation_dto.target_pub_type()] - text_measures['LEN']) / \
                mean_measures['LEN'][adaptation_dto.target_pub_type()]

    extra_sents = []
    orig_sents = com.split_into_sentences(text)

    for x in list_of_lists_of_sents:
        for y in x:
            extra_sents.append(y)
    sim_sents = similar_sentences(orig_sents, extra_sents, debug)
    while len(sim_sents) > 0 and -rel_sent_length > curr_diff:

        # Add the most similar sentence next to the original sentence
        sent = sim_sents[0]
        if debug:
            print("Appending \'" + sent[1] + "\' to \'" + sent[0] + "\'")
        text = text.replace(sent[0], sent[0] + sent[1])

        adaptation_dto.adapted_text(text)
        text_measures = adaptation_dto.text_measures()
        curr_diff = (mean_measures['LEN'][adaptation_dto.target_pub_type()] - text_measures['LEN']) / \
                mean_measures['LEN'][adaptation_dto.target_pub_type()]

        # Delete the similarity that you've already used
        sim_sents.pop(0)

    return text


def generate_additional_paragraphs(extra_content, text, debug):
    weights = list(map(lambda x: com.sentence_similarity(x, text), extra_content))
    tuples = list(zip(extra_content, weights))
    tuples.sort(key=lambda x: x[1], reverse=True)
    for x in tuples:
        text = summarize_wiki(tuples[0][0]) + "\n\n" + text
    return text


def generate(adaptation_dto, mean_measures, debug):
    text = adaptation_dto.adapted_text()
    text_measures = adaptation_dto.text_measures()

    curr_diff = (text_measures['LEN'] - mean_measures['LEN'][adaptation_dto.target_pub_type()]) / \
                mean_measures['LEN'][adaptation_dto.target_pub_type()]

    mean_sent_length = com.mean_sentence_length(text)

    rel_sent_length = (com.mean_sentence_length(adaptation_dto.orig_text())) / mean_measures['LEN'][adaptation_dto.target_pub_type()] # TO-DO

    if abs(rel_sent_length) < abs(curr_diff):
        # If we have to generate more than an average sentence, we generate extra content from DB pedia and Wikipedia
        keywords = keywords_text_rank(text)
        for keyword in keywords:
            extra_content = get_extra_content(keyword, debug)
            # https://owl.purdue.edu/owl/general_writing/academic_writing/paragraphs_and_paragraphing/paragraphing.html#:~:text=Aim%20for%20three%20to%20five,longer%20paragraphs%20for%20longer%20papers.
            if abs(rel_sent_length) * 5 < abs(curr_diff):
                # 1.) Generate whole paragraphs that are similar to the given topic
                text = generate_additional_paragraphs(extra_content, text, debug)
            else:
                # 2.) Split to sentences and find similar sentences to input after the similar sentence
                text = add_similar_sentences(adaptation_dto, mean_measures, extra_content, rel_sent_length, debug)

    else:
        pr = Preprocessing()
        pr.load_data()
        pr.encode_data()
        pr.generate_sequence()
        pr.get_data()

        # Generate text using RNN networks
        pred = Prediction(pr.tokenizer, pr.max_length)
        pred.load_model()

        sentences_desc_importance, _ = com.calc_sentence_similarity(text, True, debug)
        # We extend the least important sentence
        sentence = sentences_desc_importance[len(sentences_desc_importance) - 1]
        extended_sentence = pred.predict_sequence(sentence[1], math.floor(mean_sent_length / 2))
        text = text.replace(sentence[1], extended_sentence)
        if debug:
            print("Extended sentence: '" + extended_sentence)
    return text


def adapt_length(adaptation_dto, mean_measures, n_iterations, epsilon, debug=False):
    text = adaptation_dto.adapted_text()
    text_measures = adaptation_dto.text_measures()

    curr_diff = (text_measures['LEN'] - mean_measures['LEN'][adaptation_dto.target_pub_type()]) / \
                mean_measures['LEN'][adaptation_dto.target_pub_type()]

    if debug:
        print("Starting difference sum: ", curr_diff)

    while n_iterations > 0 and abs(curr_diff) > epsilon:
        if curr_diff > 0:
            text = summarize(adaptation_dto, mean_measures, epsilon, debug)
        else:
            text = generate(adaptation_dto, mean_measures, debug)

        length = len(tokenize.word_tokenize(text))
        curr_diff = (length - mean_measures['LEN'][adaptation_dto.target_pub_type()]) / mean_measures['LEN'][
            adaptation_dto.target_pub_type()]
        if debug:
            print("Relative difference after length manipulation: ", curr_diff)
        adaptation_dto.adapted_text(text)
        n_iterations = n_iterations - 1

    return adaptation_dto


def get_extra_content(keyword, debug):
    if debug:
        print("Keyword to generate extra text: ", keyword)

    keyword = keyword[0]

    additional_content = []

    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'opensearch',
            'format': 'json',
            'search': keyword
        }
    ).json()

    article_titles = response[1]

    for i in article_titles:
        response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': i,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
            }
        ).json()
        page = next(iter(response['query']['pages'].values()))
        additional_content.append(page['extract'])

    response = requests.get('http://lookup.dbpedia.org/api/search.asmx/KeywordSearch?QueryString=' + keyword)
    root = ET.fromstring(response.text)

    for item in root.iter():
        if item.tag == "{http://lookup.dbpedia.org/}Description":
            additional_content.append(item.text)
    return additional_content


def keywords_rake(text):
    r = Rake(
        stopwords=stopwords.words('english'),
        ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
        max_length=1
    )

    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases_with_scores()
    print(keywords)


def keywords_text_rank(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text, candidate_pos=['NOUN', 'PROPN'], window_size=4, lower=False)
    return tr4w.get_keywords(10)

def summarize_wiki(text):
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cpu')

    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_text = "summarize: " + preprocess_text

    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)

    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=70,
                                 max_length=130,
                                 early_stopping=False,
                                 repetition_penalty=2.5,
                                 length_penalty=1.50)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=False)

    return summary