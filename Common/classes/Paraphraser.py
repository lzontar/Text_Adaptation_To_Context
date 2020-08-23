import string

import nltk
import re
import pattern.text.en as en
import mlconjug

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import Common.library.Common as com

class Paraphraser:
    def __init__(self, text, debug=False):
        self._text = text
        self._tokens_original = None
        self._tokens_norm = None
        self._freq_original = None
        self._freq_norm = None
    def preprocess(self):
        self._text = self._text.lower()
        self._text = re.sub(r'\d+', '', self._text)
        self._text = re.sub('[^A-Za-z0-9]+', ' ', self._text)
        self._text = self._text.strip()
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = nltk.tokenize.word_tokenize(self._text)
        tokens_without_stopwords = list(filter(lambda x: x not in stop_words, tokens))

        self._tokens_norm = []

        stemmer = nltk.stem.PorterStemmer()
        lemmatizer = nltk.stem.WordNetLemmatizer()

        for word in tokens_without_stopwords:
            self._tokens_norm.append(lemmatizer.lemmatize(word))
            print(stemmer.stem(word))
            print(lemmatizer.lemmatize(word))

        # self._tokens_original = nltk.ne_chunk(nltk.pos_tag(self._tokens_original))
        # self._tokens_norm = nltk.ne_chunk(nltk.pos_tag(self._tokens_norm))


        self._freq_original = nltk.FreqDist(self._tokens_original)
        self._freq_norm = nltk.FreqDist(self._tokens_norm)

    def paraphrase_sentence(self):
        def set_seed(seed):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        set_seed(42)

        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device ", device)
        model = model.to(device)

        sentence = "here was once a velveteen rabbit, and in the beginning he was really splendid."
        # sentence = "What are the ingredients required to bake a perfect cake?"
        # sentence = "What is the best possible approach to learn aeronautical engineering?"
        # sentence = "Do apples taste better than oranges in general?"

        text = "paraphrase: " + sentence + " </s>"

        max_len = 256

        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=10
        )

        print("\nOriginal Question ::")
        print(sentence)
        print("\n")
        print("Paraphrased Questions :: ")
        final_outputs = []
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)

    def try_fix_form(word_pos, syn_pos):
        word = word_pos[0]
        syn = syn_pos[0]
        pos_tag_word = word_pos[1]
        pos_tag_syn = syn_pos[1]

        if pos_tag_syn != pos_tag_word:
            # Check if its only plural version
            if pos_tag_word == pos_tag_syn + 'S':
                if pos_tag_syn.startswith('J'):
                    return en.superlative(syn)
                elif pos_tag_syn.startswith('N'):
                    return en.pluralize(syn)
            return None if pos_tag_syn[:2] != pos_tag_word[:2] else syn
        else:
            if not pos_tag_syn.startswith('V'):
                return syn
            # We check if verb is in correct form
            default_conjugator = mlconjug.Conjugator(language='en')

            if pos_tag_word == 'VB':
                return default_conjugator.conjugate(syn).conjug_info['indicative']['indicative present']['1s']
            elif pos_tag_word == 'VBG':
                return default_conjugator.conjugate(syn).conjug_info['indicative']['indicative present continuous']['1s 1s']
            elif pos_tag_word == 'VBN':
                return default_conjugator.conjugate(syn).conjug_info['indicative']['indicative present perfect']['1p']
            elif pos_tag_word == 'VBP':
                return default_conjugator.conjugate(syn).conjug_info['indicative']['indicative present']['1s']
            elif pos_tag_word == 'VBZ':
                return default_conjugator.conjugate(syn).conjug_info['indicative']['indicative present']['3s']
            elif pos_tag_word == 'VBD':
                return default_conjugator.conjugate(syn).conjug_info['indicative']['indicative past tense']['1s']
