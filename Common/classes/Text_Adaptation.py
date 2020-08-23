import Text_Characteristics

import Length_Manipulator.Summarizer as lm
import Length_Manipulator.Summarizer_NN as sum_nn

import Synonym_Replacer.Synonym_Replacer as sr
import Synonym_Replacer.Paraphraser as p

import Common.library.Common as com
from Common.classes.AdaptationDTO import AdaptationDTO
import operator


def adapt_text(mode, model_para, model_gen, model_sum, tokenizer_para, tokenizer_gen, tokenizer_sum, device_para, device_gen, device_sum, text, mean_measures, target_publication_type, nlp, dictionary, tc,debug=False):
    debug = False
    epsilon = 0.1
    n_iterations_len = 3
    adaptation_dto = AdaptationDTO(text, target_pub_type=target_publication_type, debug=debug)

    text_after_first_length_manipulation = None
    for i in range(n_iterations_len):
        # adaptation_dto = p.adapt_complexity_and_polarity(model_para, tokenizer_para, device_para, adaptation_dto, mean_measures, n_iterations, epsilon, debug)
        # adaptation_dto = lm.adapt_length(adaptation_dto, mean_measures, n_iterations, epsilon, debug)

        adaptation_dto = sum_nn.adapt_length(model_gen, model_sum, tokenizer_gen, tokenizer_sum, device_gen, device_sum, adaptation_dto, mean_measures, 1, epsilon, debug)
        n_iterations = round(100 / n_iterations_len)
        if target_publication_type == 'RES_ARTCL':
            n_iterations = round(250 / n_iterations_len)
        if target_publication_type == 'SOC_MED':
            n_iterations = round(15 / n_iterations_len)
        if i == 0:
            text_after_length_manipulation = adaptation_dto.adapted_text()
        adaptation_dto = sr.adapt_complexity_and_polarity(adaptation_dto, mean_measures, n_iterations, epsilon, nlp, dictionary, tc, debug)

        n_iterations = round(20 / n_iterations_len)
        if target_publication_type == 'RES_ARTCL':
            n_iterations = round(50 / n_iterations_len)
        if target_publication_type == 'SOC_MED':
            n_iterations = round(3 / n_iterations_len)
        if mode == 'PARA':
            adaptation_dto = p.adapt_complexity_and_polarity(model_para, tokenizer_para, device_para, adaptation_dto,
                                                             mean_measures, n_iterations, epsilon, tc, debug)
        beautified = com.beautify_text(adaptation_dto.adapted_text(), adaptation_dto, debug)
        adaptation_dto.adapted_text(beautified)

    # if debug:
    #     print("\n\n    ------------------------------- ORIGINAL TEXT -------------------------------    \n\n",
    #           text)
    #     print("\n\n    ------------------------------- ADAPTED  TEXT -------------------------------    \n\n",
    #           beautified)

    return beautified, text_after_length_manipulation
