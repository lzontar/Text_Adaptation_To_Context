import Text_Characteristics

import Length_Manipulator.Summarizer as lm
import Length_Manipulator.Summarizer_NN as sum_nn

import Synonym_Replacer.Synonym_Replacer as sr
import Synonym_Replacer.Paraphraser as p

import Common.library.Common as com
from Common.classes.AdaptationDTO import AdaptationDTO
import operator


def adapt_text(text, mean_measures, target_publication_type, debug=False):
    debug = False
    epsilon = 0.1
    n_iterations = 100

    adaptation_dto = AdaptationDTO(text, target_pub_type=target_publication_type, debug=debug)
    adaptation_dto = sum_nn.adapt_length(adaptation_dto, mean_measures, n_iterations, epsilon)
    adaptation_dto = p.adapt_complexity_and_polarity(adaptation_dto, mean_measures, n_iterations, epsilon)
    # adaptation_dto = lm.adapt_length(adaptation_dto, mean_measures, n_iterations, epsilon, debug)
    #
    # adaptation_dto = sr.adapt_complexity_and_polarity(adaptation_dto, mean_measures, n_iterations, epsilon, debug)

    beautified = com.beautify_text(adaptation_dto.adapted_text(), adaptation_dto, debug)

    if debug:
        print("\n\n    ------------------------------- ORIGINAL TEXT -------------------------------    \n\n",
              text)
        print("\n\n    ------------------------------- ADAPTED  TEXT -------------------------------    \n\n",
              beautified)

    return beautified
