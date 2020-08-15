import random
import statistics
from collections import defaultdict

import pandas as pd
import textstat
import os.path
import nltk

from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import Common.library.Common as com
import json


class Text_Characteristics:
    _metrics_map = defaultdict(list)
    _variance_map = defaultdict(list)

    _lang = 'en_US'

    _data_absolute_path = 'C:/Luka/School/Bachelor/Bachelor\'s thesis/Text_Adaptation/Data/covid-19/'

    def __init__(self, target_pub_type=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        nltk.download('vader_lexicon')
        textstat.set_lang(self._lang)
        self._metrics_map['LEN'] = defaultdict(list)  # Length of the origin text
        self._metrics_map['SENT_ANAL'] = defaultdict(list)  # Sentiment analysis
        self._metrics_map['READ'] = defaultdict(list)  # Readability

        self._variance_map['LEN'] = defaultdict(list)  # Length of the origin text
        self._variance_map['SENT_ANAL'] = defaultdict(list)  # Sentiment analysis
        self._variance_map['READ'] = defaultdict(list)  # Readability

        self._target_type = target_pub_type

    def mean_measures(self, debug=False):
        self.calc_means(debug)
        return self._metrics_map

    def calc_means(self, debug):

        ''' Calculate mean values for social media / blogs '''
        if self._orig_type == 'SOC_MED' or self._target_type == 'SOC_MED' or self._target_type is None or self._orig_type is None:
            if debug:
                print('Calculating means for social media...')
            self.calc_means_SOC_MED(debug)
            if debug:
                print('means for social media are calculated')

        ''' Calculate mean values for newspapers '''
        if self._orig_type == 'NEWS' or self._target_type == 'NEWS' or self._target_type is None or self._orig_type is None:
            if debug:
                print('Calculating means for news...')
            self.calc_means_NEWS(debug)
            if debug:
                print('means for news are calculated')

        ''' Calculate mean values for research articles '''
        if self._orig_type == 'RES_ARTCL' or self._target_type == 'RES_ARTCL' or self._target_type is None or self._orig_type is None:
            if debug:
                print('Calculating means for research articles...')
            self.calc_means_RES_ARTCL(debug)
            if debug:
                print('means for research articles are calculated')
        ''' Calculate mean values for official statements '''
        if self._orig_type == 'OFC_STMT' or self._target_type == 'OFC_STMT' or self._target_type is None or self._orig_type is None:
            if debug:
                print('Calculating means for official statements...')
            self.calc_means_OFC_STMT(debug)
            if debug:
                print('means for official statements are calculated')

    def calc_means_SOC_MED(self, debug):
        try:
            # dataset = pd.read_csv(self._data_absolute_path + 'tweets/blogs.csv')['text'].iloc[:10000]
            dataset = random.sample(set(pd.read_csv(self._data_absolute_path + 'tweets/covid19_tweets.csv')['text']), 150)
            # LEN: 32113
        except IOError as e:
            print(e)
            raise
        publication_type = 'SOC_MED'
        length = self.calc_length_mean(dataset, publication_type, debug)
        sentiment_analysis = self.calc_sentiment_analysis_mean(dataset, publication_type, debug)
        readability = self.calc_readability_mean(dataset, publication_type, debug)
        return length, sentiment_analysis, readability

    def calc_means_NEWS(self, debug):
        try:
            dataset = random.sample(set(pd.read_csv(self._data_absolute_path + 'news/news.csv')['text']), 150)
            # LEN: 6788

            # dataset = []
            # exist_files = True
            # i = 0
            # while exist_files:
            #     exist_files = False
            #     filepath = self._data_absolute_path + 'bbc/business/' + format(i + 1, '03') + '.txt'
            #     if os.path.isfile(filepath):
            #         file = open(filepath, 'r')
            #         dataset.append(file.read())
            #         exist_files = True
            #
            #     filepath = self._data_absolute_path + 'bbc/entertainment/' + format(i + 1, '03') + '.txt'
            #     if os.path.isfile(filepath):
            #         file = open(filepath, 'r')
            #         dataset.append(file.read())
            #         exist_files = True
            #
            #     filepath = self._data_absolute_path + 'bbc/politics/' + format(i + 1, '03') + '.txt'
            #     if os.path.isfile(filepath):
            #         file = open(filepath, 'r')
            #         dataset.append(file.read())
            #         exist_files = True
            #
            #     filepath = self._data_absolute_path + 'bbc/sport/' + format(i + 1, '03') + '.txt'
            #     if os.path.isfile(filepath):
            #         file = open(filepath, 'r')
            #         dataset.append(file.read())
            #         exist_files = True
            #
            #     filepath = self._data_absolute_path + 'bbc/tech/' + format(i + 1, '03') + '.txt'
            #     if os.path.isfile(filepath):
            #         file = open(filepath, 'r')
            #         dataset.append(file.read())
            #         exist_files = True
            #
            #     i = i + 1

            if debug:
                'Dataset was successfully read from the file'
        except FileNotFoundError:
            print('File was not found, it should be located at: "bbc/{GENRE}/"')
            raise
        publication_type = 'NEWS'
        length = self.calc_length_mean(dataset, publication_type, debug)
        sentiment_analysis = self.calc_sentiment_analysis_mean(dataset, publication_type, debug)
        readability = self.calc_readability_mean(dataset, publication_type, debug)
        return length, sentiment_analysis, readability

    def calc_means_RES_ARTCL(self, debug):
        try:
            # dataset = pd.read_csv(self._data_absolute_path + 'research_articles/abstracts.csv')['abstract'].iloc[:10000]
            rootdir = self._data_absolute_path + 'research_articles/document_parses/pdf_json'
            dataset = []
            for subdir, dirs, files in os.walk(rootdir):
                ix = 0
                for f in files:
                    if ix == 150:
                        break
                    file = open(self._data_absolute_path + 'research_articles/document_parses/pdf_json/' + f, 'r')
                    json_file = json.loads(str(file.read()))
                    research_article = ''
                    for paragraph in json_file['body_text']:
                        research_article = research_article + paragraph['text']

                    file.close()
                    dataset.append(research_article)
                    ix = ix + 1
        except IOError as e:
            print(e)
        publication_type = 'RES_ARTCL'
        length = self.calc_length_mean(dataset, publication_type, debug)
        sentiment_analysis = self.calc_sentiment_analysis_mean(dataset, publication_type, debug)
        readability = self.calc_readability_mean(dataset, publication_type, debug)
        return length, sentiment_analysis, readability

    def calc_means_OFC_STMT(self, debug):
        # file = open(self._data_absolute_path + 'official_statements/combined.json', 'r')
        # data_json = file.read().splitlines()
        # dataset = []
        dataset = random.sample(set(pd.read_csv(self._data_absolute_path + 'official_statements/data.csv')['content']), 150)
        # LEN: 156
        publication_type = 'OFC_STMT'
        length = self.calc_length_mean(dataset, publication_type, debug)
        sentiment_analysis = self.calc_sentiment_analysis_mean(dataset, publication_type, debug)
        readability = self.calc_readability_mean(dataset, publication_type, debug)
        return length, sentiment_analysis, readability

    def calc_length_mean(self, dataset, publication_type, debug):
        lengths = []

        for i in range(len(dataset)):
            if type(dataset[i]) is not float:
                lengths.append(len(tokenize.word_tokenize(dataset[i])))

        length_mean = statistics.mean(lengths)
        length_variance = statistics.stdev(lengths)
        if publication_type is not None:
            self._metrics_map['LEN'][publication_type] = length_mean
            self._variance_map['LEN'][publication_type] = length_variance

        if debug:
            print('mean length: ' + length_mean.__str__() + ' tokens')

        return length_mean

    def calc_polarity_scores(self, text):
        sentences = tokenize.sent_tokenize(text)
        sentence_polarity = []

        for s in sentences:
            sentiment_score = self.sentiment_analyzer.polarity_scores(s)['compound']
            sentence_polarity.append(sentiment_score)
        return statistics.mean(sentence_polarity)

    def calc_sentiment_analysis_mean(self, dataset, publication_type, debug):

        polarity = []

        for i in range(len(dataset)):
            if type(dataset[i]) is not float:
                sentences = tokenize.sent_tokenize(dataset[i])
                sentence_polarity = []
                for s in sentences:
                    sentiment_score = self.sentiment_analyzer.polarity_scores(s)
                    sentence_polarity.append(sentiment_score['compound'])

                if len(sentence_polarity) > 0:  # and len(sentence_subjectivity) > 0:
                    polarity.append(statistics.mean(sentence_polarity))

        polarity_mean = statistics.mean(polarity)
        sentiment_scores_mean = {
            'POLAR': polarity_mean,
        }
        polarity_variance = statistics.stdev(polarity)

        if publication_type is not None:
            self._metrics_map['SENT_ANAL'][publication_type] = sentiment_scores_mean
            self._variance_map['SENT_ANAL'][publication_type] = polarity_variance

        if debug:
            print('mean polarity: ' + sentiment_scores_mean['POLAR'].__str__())

        return sentiment_scores_mean

    def calc_readability_mean(self, dataset, publication_type, debug):
        readability_measures = []

        for i in range(len(dataset)):
            if type(dataset[i]) is not float:
                readability_measures.append(com.flesch_reading_ease(dataset[i]))

        readability_mean = statistics.mean(readability_measures)
        readability_variance = statistics.stdev(readability_measures)

        if publication_type is not None:
            self._metrics_map['READ'][publication_type] = readability_mean
            self._variance_map['READ'][publication_type] = readability_variance

        if debug:
            print('mean readability: ' + readability_mean.__str__())

        return readability_mean

    def calc_text_measures(self, text, debug=False):
        sentences = tokenize.sent_tokenize(text)
        sentence_polarity = []
        sentence_subjectivity = []
        for s in sentences:
            sentiment_score = self.sentiment_analyzer.polarity_scores(s)['compound']
            sentence_polarity.append(sentiment_score)

        return {
            "LEN": len(tokenize.word_tokenize(text)),
            "SENT_ANAL": {
                "POLAR": statistics.mean(sentence_polarity)
            },
            "READ": com.flesch_reading_ease(text)
        }

    def variance_map(self):
        return self._variance_map
