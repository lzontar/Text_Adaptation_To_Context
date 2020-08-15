import spacy
from nltk import Tree


def tok_format(node):
    return "_".join([node.orth_, node.tag_])


def to_nltk_print(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_print(child) for child in node.children])
    else:
        return tok_format(node)


class SentenceTree:
    def __init__(self, sentence, debug):
        """ Constructor for this class. """
        self.nlp = spacy.load('en_core_web_sm')

        self.tree = self.nlp(sentence[1])
        if debug:
            [sent if type(to_nltk_print(sent.root)) is str else to_nltk_print(sent.root).pretty_print() for sent in self.tree.sents]
