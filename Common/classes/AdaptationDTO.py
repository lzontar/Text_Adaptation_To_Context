import Common.library.Common as com
from Common.classes.AdaptationGraph import AdaptationGraph
import Text_Characteristics.Text_Characteristics as tc


class AdaptationDTO:
    def __init__(self, text, target_pub_type, debug=False):
        """ Constructor for this class. """
        self._orig_text = text
        self._adapted_text = text.lower()
        self._target_pub_type = target_pub_type

        self._entities, self._lower_entities = com.entities_from_text(text)
        self.generate_adapted_text(debug)
        self.debug = debug

    def build_mapper_orig_to_importance(self, debug):
        mapper = []
        for i in range(len(self._orig_ordered_sentences)):
            for j in range(len(self._importance_ordered_sentences)):
                if self._orig_ordered_sentences[i] == self._importance_ordered_sentences[j]:
                    mapper.append((i, j))
        if debug:
            print(mapper)
        return mapper

    def generate_adapted_text(self, debug):
        # self._graph = AdaptationGraph(self.adapted_text(), debug)
        # self._orig_ordered_sentences = com.split_into_sentences(self.adapted_text())
        # self._importance_ordered_sentences, _ = com.calc_sentence_similarity(self.adapted_text(), debug=debug)
        # self._orig_to_importance_sentence_mapper = self.build_mapper_orig_to_importance(debug)
        text_characteristics = tc(self.target_pub_type())
        self._text_measures = text_characteristics.calc_text_measures(self.adapted_text())

    def orig_text(self):
        return self._orig_text

    def adapted_text(self, value=None):
        if value is not None:
            self._adapted_text = value
            self.generate_adapted_text(self.debug)
        return self._adapted_text

    def target_pub_type(self, value=None):
        if value is not None:
            self._target_pub_type = value
        return self._target_pub_type

    def text_measures(self):
        return self._text_measures

    # def graph(self):
    #     return self._graph

    def entities(self):
        return self._entities, self._lower_entities
