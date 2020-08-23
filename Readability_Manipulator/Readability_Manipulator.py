import nltk
import syllables
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import textstat
import Text_Characteristics.Text_Characteristics as tc
from PyDictionary import PyDictionary
import mlconjug
import spacy
import Common.library.Common as com

def relative_difference_dict_item(x, mean_measures, adaptation_dto):
    readability = abs((x['READ'] - mean_measures['READ'][adaptation_dto.target_pub_type()]) / \
               mean_measures['READ'][adaptation_dto.target_pub_type()])
    return {
        'WORD': x['WORD'],
        'READ': readability,
        'TAG': x['TAG']
    }


def word_dictionary_item(x):
    word = x.text
    tag = x.tag_
    return {
        'WORD': word,
        'READ': syllables.estimate(word),
        'TAG': tag
    }


def adapt_readability(adaptation_dto, mean_measures, n_iterations, epsilon, debug):
    dictionary = PyDictionary()
    nlp = spacy.load("en_core_web_sm")

    text = adaptation_dto.adapted_text()

    doc = nlp(text)

    _, entities = adaptation_dto.entities()

    stop_words = set(stopwords.words('english'))

    regex = re.compile('[^a-zA-Z]')
    filtered_tokens = set(
        [w for w in doc if w.text not in stop_words and w.text == regex.sub('', w.text) and w.text not in entities])

    readability = list(map(word_dictionary_item, list(filtered_tokens)))

    rel_readability = list(map(lambda x: relative_difference_dict_item(x, mean_measures, adaptation_dto),
                                        readability))

    sorted_rel_readability = sorted(rel_readability, key=lambda k: k['READ'],
                                             reverse=True)

    n_words = len(filtered_tokens)
    rel_read = abs((adaptation_dto.text_measures()['READ'] - mean_measures['READ'][adaptation_dto.target_pub_type()]) / mean_measures['READ'][adaptation_dto.target_pub_type()])

    curr_diff = rel_read
    if debug:
        print("Starting difference sum: ", curr_diff)

    ix = 0
    text_characteristics = tc(adaptation_dto.target_pub_type())

    while ix < n_iterations and ix < n_words and abs(curr_diff) > epsilon:
        word = sorted_rel_readability[ix]['WORD']
        tag = sorted_rel_readability[ix]['TAG']
        best_synonym = None
        best_synonym_diff = None
        best_synonym_read = None

        synonyms = dictionary.synonym(word)

        for syn in (synonyms if synonyms is not None else []):
            syn = try_fix_form((word, tag), nltk.pos_tag([syn])[0])
            if syn is None:
                continue

            text_with_syn = text

            text_with_syn = re.sub(r'\b' + word + '\b', syn, text_with_syn)
            text_with_syn = re.sub(r'([^a-zA-Z]+)' + word + r'([^a-zA-Z]+)', r'\1' + syn + r'\2', text_with_syn)

            curr_read_with_syn = com.flesch_reading_ease(text_with_syn)

            rel_read_with_syn = abs((
                curr_read_with_syn - mean_measures['READ'][adaptation_dto.target_pub_type()]) / \
                                 mean_measures['READ'][adaptation_dto.target_pub_type()])

            curr_diff_with_syn = rel_read_with_syn

            if best_synonym is None or (curr_diff_with_syn < best_synonym_diff):
                best_synonym = syn
                best_synonym_diff = curr_diff_with_syn
                best_synonym_read = curr_read_with_syn

        if best_synonym is not None and best_synonym != word and curr_diff > best_synonym_diff:
            # We change all occurrences and set new current readability
            text = re.sub(r'\b' + word + '\b', best_synonym, text)
            text = re.sub(r'([^a-zA-Z]+)' + word + r'([^a-zA-Z]+)', r'\1' + best_synonym + r'\2', text)

            if debug:
                print("Replacing '", word, "' for '", best_synonym, "'")
                print("Relative difference after replacement: ", best_synonym_diff)
                print("Readability after replacement: ", best_synonym_read)

            curr_diff = best_synonym_diff

        ix = ix + 1

    adaptation_dto.adapted_text(text)

    return adaptation_dto


def try_fix_form(word_pos, syn_pos):
    word = word_pos[0]
    syn = syn_pos[0]
    pos_tag_word = word_pos[1]
    pos_tag_syn = syn_pos[1]

    if pos_tag_syn != pos_tag_word:
        # Check if its only plural version
        if pos_tag_word == pos_tag_syn + 'S':
            if pos_tag_syn.startswith('J'):
                return syn
            elif pos_tag_syn.startswith('N'):
                return syn
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
