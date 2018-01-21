from typing import List
import re


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from re import fullmatch, match

def get_shape_category_simple(word):
    if word.islower():
        return 'ALL-LOWER'
    elif word.isupper():
        return 'ALL-UPPER'
    elif fullmatch('[A-Z][a-z]+', word):
        return 'FIRST-UPPER'
    else:
        return 'MISC'


def get_shape_category(token):
    if match('^[\n]+$', token):  # IS LINE BREAK
        return 'NL'
    if any(char.isdigit() for char in token) and match('^[0-9.,]+$', token):  # IS NUMBER (E.G., 2, 2.000)
        return 'NUMBER'
    if fullmatch('[^A-Za-z0-9\t\n ]+', token):  # IS SPECIAL CHARS (E.G., $, #, ., *)
        return 'SPECIAL'
    if fullmatch('^[A-Z\-.]+$', token):  # IS UPPERCASE (E.G., AGREEMENT, INC.)
        return 'ALL-CAPS'
    if fullmatch('^[A-Z][a-z\-.]+$', token):  # FIRST LETTER UPPERCASE (E.G. This, Agreement)
        return '1ST-CAP'
    if fullmatch('^[a-z\-.]+$', token):  # IS LOWERCASE (E.G., may, third-party)
        return 'LOWER'
    if not token.isupper() and not token.islower():  # WEIRD CASE (E.G., 3RD, E2, iPhone)
        return 'MISC'
    return 'MISC'


class Interval:
    """A class for representing a contiguous range of integers"""

    def __init__(self, start: int, end: int):
        """
        :param start: start of the range
        :param end: first integer not included the range
        """
        self.start = int(start)
        self.end = int(end)
        if self.start > self.end:
            raise ValueError('Start "{}" must not be greater than end "{}"'.format(self.start, self.end))
        if self.start < 0:
            raise ValueError('Start "{}" must not be negative'.format(self.start))

    def __len__(self):
        """ Return end - start """
        return self.end - self.start

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return self.start != other.start or self.end != other.end

    def __lt__(self, other):
        return (self.start, -len(self)) < (other.start, -len(other))

    def __le__(self, other):
        return (self.start, -len(self)) <= (other.start, -len(other))

    def __gt__(self, other):
        return (self.start, -len(self)) > (other.start, -len(other))

    def __ge__(self, other):
        return (self.start, -len(self)) >= (other.start, -len(other))

    def __hash__(self):
        return hash(tuple(v for k, v in sorted(self.__dict__.items())))

    def __contains__(self, item: int):
        """ Return self.start <= item < self.end """
        return self.start <= item < self.end

    def __repr__(self):
        return 'Interval[{}, {}]'.format(self.start, self.end)

    def __str__(self):
        return repr(self)

    def intersection(self, other) -> 'Interval':
        """ Return the interval common to self and other """
        a, b = sorted((self, other))
        if a.end <= b.start:
            return Interval(self.start, self.start)
        return Interval(b.start, min(a.end, b.end))

    def overlaps(self, other) -> bool:
        """ Return True if there exists an interval common to self and other """
        a, b = sorted((self, other))
        return a.end > b.start

    def shift(self, i: int):
        self.start += i
        self.end += i


class Token(Interval):
    """ A Interval representing word like units of text with a dictionary of features """

    def __init__(self, document, start: int, end: int, shape: int, text: str,label: str=None):
        """
        Note that a token has 2 text representations.
        1) How the text appears in the original document e.g. doc.text[token.start:token.end]
        2) How the tokeniser represents the token e.g. nltk.word_tokenize('"') == ['``']
        :param document: the document object containing the token
        :param start: start of token in document text
        :param end: end of token in document text
        :param pos: part of speach of the token
        :param shape: integer label describing the shape of the token
        :param text: this is the text representation of token
        """

        Interval.__init__(self, start, end)
        self._doc = document
        self._label = label
        self._shape = shape
        self._text = text


    @property
    def text(self):
        return self._text

    @property
    def pos(self):
        return self._pos

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, item):
        return self._text[item]

    def __repr__(self):
        return 'Token({}, {}, {})'.format(self.text, self.start, self.end)


class Sentence(Interval):
    """ Interval corresponding to a Sentence"""

    def __init__(self, document, start: int, end: int):
        Interval.__init__(self, start, end)
        self._doc = document

    def __repr__(self):
        return 'Sentence({}, {})'.format(self.start, self.end)

    @property
    def tokens(self):
        """Returns the list of tokens contained in a sentence"""
        return [token for token in self._doc.tokens if token.overlaps(self)]


class Document:
    """
    A document is a combination of text and the positions of the tags and elements in that text.
    """
    def __init__(self):
        self.text = None
        self.tokens = None
        self.sentences = None


    @classmethod
    def create_from_text(cls,  text: str=None):
        """
        :param text: document text as a string
        """
        doc = Document()
        doc.text = text
        # TODO: To be implemented
        # 1. Tokenize texte (tokens & phrases)
        if text != "":
            words, pos_tags = zip(*nltk.pos_tag(nltk.word_tokenize(text)))
            sentences = nltk.sent_tokenize(text.replace('\n', ' '))
            # 2. Corriger la tokenisation (retokenize)
            words, pos_tags = Document._retokenize(words, pos_tags)
            # 3. Trouver les intervalles de Tokens
            doc.tokens = Document._find_tokens(doc, words, pos_tags, text)
            # 4. Trouver les intervalles de phrases
            doc.sentences = Document._find_sentences(text, sentences)
        
        return doc


    @staticmethod
    def _retokenize(word_tokens: List[str], pos_tags: List[str]):
        """
        Correct NLTK tokenization. We separate symbols from words, such as quotes, -, *, etc
        :param word_tokens: list of strings(tokens) coming out of nltk.word_tokenize
        :return: new list of tokens
        """
        split_end = re.escape('-*.')
        split_end_chars = '-*.'
        always_split = re.escape('’`"\'“”/\\')
        always_split_chars = '’`"\'“”/\\'
        to_process = re.compile("[-*.]|[’`\"\'“\/]")
        new_tokens, new_pos = [], []
        for t, p in zip(word_tokens, pos_tags):
            if to_process.findall(t) != None:
                split_tokens = re.split('([' + always_split + ']+)|(\n)|(^[' + split_end + '])|([' + split_end + ']$)', t)
                split_tokens = [t for t in split_tokens if t is not None and t != '']
                new_tokens.extend(split_tokens)
                for sp in split_tokens:
                    if any(True if c in sp else False for c in always_split_chars) or any(
                            True if c in sp else False for c in split_end_chars):
                        new_pos.append(sp)
                    else:
                        new_pos.append(p)
            else:
                new_tokens.extend(t)
                new_pos.extend(p)
        return new_tokens, new_pos

    @staticmethod
    def _find_tokens(doc, word_tokens, pos_tags, text):
        """ Calculate the span of each token,
         find which element it belongs to and create a new Token instance """
        offset = 0
        tokens = []
        missing = None
        for token, pos_tag in zip(word_tokens, pos_tags):
            position = text.find(token, offset, offset + max(len(token), 50))
            if position > -1:
                if missing:
                    continue
                else:
                    tokens.append(Token(document = doc, start = position, end = position + len(token),shape="",text=token, label = pos_tag))
                    offset = position + len(token)
            else:
                continue
        return tokens
    

    @staticmethod
    def _find_sentences(doc_text: str, sentences):
        """ yield Sentence objects each time a sentence is found in the text """
        offset = 0
        inter = list()
        for sentence in sentences:
            position = doc_text.find(sentence, offset)
            if position > -1:
                inter.append(Interval(start=position, end=position + len(sentence)))
                offset = position + len(sentence)
            else:
                continue
        return inter