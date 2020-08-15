import Common.library.Common as com
from Common.classes.SentenceTree import SentenceTree


class AdaptationGraph:
    def __init__(self, text, debug):
        """ Constructor for this class. """
        self.edges = {}

        sentences, similarity_graph = com.calc_sentence_similarity(text, debug=debug)
        self.nodes = []
        for s in sentences:
            self.nodes.append(SentenceTree(s, debug))

        self.build_graph(sentences, similarity_graph, debug)

    def build_graph(self, sentences, similarity_graph, debug):
        similarity_array = similarity_graph.toarray()
        for i in range(len(similarity_array)):
            for j in range(len(similarity_array[i])):
                if i != j and not (self.edges.keys().__contains__((i, j)) or self.edges.keys().__contains__((j, i))):
                    self.edges[(i, j)] = similarity_array[i][j]

