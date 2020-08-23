import nltk
import syllables
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import textstat
import Text_Characteristics.Text_Characteristics as tc
from PyDictionary import PyDictionary
import pattern.text.en as en
import mlconjug
import spacy
import Common.library.Common as com

def relative_difference_dict_item(x, mean_measures, adaptation_dto):
    polarity = abs((x['POLAR'] - mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR']) / \
               mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR'])
    readability = abs((x['READ'] - mean_measures['READ'][adaptation_dto.target_pub_type()]) / mean_measures['READ'][
        adaptation_dto.target_pub_type()])
    return {
        'WORD': x['WORD'],
        'POLAR': polarity,
        'READ': readability,
        'RP': polarity + readability,
        'TAG': x['TAG']
    }


def word_dictionary_item(x):
    word = x.text
    tag = x.tag_
    return {
        'WORD': word,
        'POLAR': SentimentIntensityAnalyzer().polarity_scores(word)['compound'],
        'READ': syllables.estimate(word),
        'TAG': tag
    }


def adapt_complexity_and_polarity(adaptation_dto, mean_measures, n_iterations, epsilon, nlp, dictionary, text_characteristics, debug):

    text = adaptation_dto.adapted_text()

    doc = nlp(text)

    _, entities = adaptation_dto.entities()

    stop_words = set(stopwords.words('english'))

    regex = re.compile('[^a-zA-Z]')
    filtered_tokens = set(
        [w for w in doc if w.text not in stop_words and w.text == regex.sub('', w.text) and w.text not in entities])

    polarity_readability = list(map(word_dictionary_item, list(filtered_tokens)))

    rel_polarity_readability = list(map(lambda x: relative_difference_dict_item(x, mean_measures, adaptation_dto),
                                        polarity_readability))

    sorted_rel_polarity_readability = sorted(rel_polarity_readability, key=lambda k: k['RP'],
                                             reverse=True)

    n_words = len(filtered_tokens)
    rel_polar = abs((adaptation_dto.text_measures()['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR']) / mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR'])
    rel_read = abs((
        adaptation_dto.text_measures()['READ'] - mean_measures['READ'][adaptation_dto.target_pub_type()]) / \
               mean_measures['READ'][adaptation_dto.target_pub_type()])

    curr_diff = rel_polar + rel_read
    if debug:
        print("Starting difference sum: ", curr_diff)

    ix = 0
    while ix < n_iterations and ix < n_words and abs(curr_diff) > epsilon:
        word = sorted_rel_polarity_readability[ix]['WORD']
        tag = sorted_rel_polarity_readability[ix]['TAG']
        best_synonym = None
        best_synonym_diff = None

        synonyms = dictionary.synonym(word)
        if synonyms is None:
            synonyms = dictionary.synonym(word.lower())

        for syn in (synonyms if synonyms is not None else []):
            syn = try_fix_form((word, tag), nltk.pos_tag([syn])[0])
            if syn is None:
                continue

            text_with_syn = text
            if re.search('[^a-zA-Z]', syn) is None:
                text_with_syn = re.sub(r'([^\w]+)' + word + r'([^\w]+)', r'\1' + syn + r'\2', text_with_syn)
                text_with_syn = re.sub(r'([^\w]+)' + word.lower() + r'([^\w]+)', r'\1' + syn + r'\2', text_with_syn)
            else:
                text_with_syn = text_with_syn.replace(word, syn)
                text_with_syn = text_with_syn.replace(word.lower(), syn)

            curr_polar_with_syn = text_characteristics.calc_polarity_scores(text_with_syn)
            curr_read_with_syn = com.flesch_reading_ease(text_with_syn)

            rel_polar_with_syn = abs((
                curr_polar_with_syn - mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR']) / \
                                 mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR'])
            rel_read_with_syn = abs((curr_read_with_syn - mean_measures['READ'][adaptation_dto.target_pub_type()]) / \
                                mean_measures['READ'][adaptation_dto.target_pub_type()])

            curr_diff_with_syn = rel_polar_with_syn + rel_read_with_syn

            if best_synonym is None or (curr_diff_with_syn < best_synonym_diff):
                best_synonym = syn
                best_synonym_diff = curr_diff_with_syn

        if best_synonym is not None and best_synonym != word and curr_diff > best_synonym_diff:
            # We change all occurrences and set new current polarity

            if re.search('[^a-zA-Z]', best_synonym) is None:
                text = re.sub(r'([^\w]+)' + word + r'([^\w]+)', r'\1' + best_synonym + r'\2', text_with_syn)
                text = re.sub(r'([^\w]+)' + word.lower() + r'([^\w]+)', r'\1' + best_synonym + r'\2', text_with_syn)
            else:
                text = text_with_syn.replace(word, best_synonym)
                text = text_with_syn.replace(word.lower(), best_synonym)

            if debug:
                print("Replacing '", word, "' for '", best_synonym, "'")
                print("Relative difference after replacement: ", best_synonym_diff)
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
