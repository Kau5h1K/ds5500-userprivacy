import pandas as pd
import numpy as np
import glob
import warnings

warnings.filterwarnings('ignore')
import os

import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.utils import tokenize
from gensim import utils
from gensim.models import FastText
from nltk.tokenize import word_tokenize
import time
import tempfile
import os

import time

_start_time = time.time()

def tic():
    global _start_time
    _start_time = time.time()

def toc():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))


def cleanText(text, lower=True, stem=False, remove_stopwords=False, isolate_sym = True, remove_alphanum = False):
    """
    Function to clean the segment text
    :param text: text to process
    :param lower: lower the text
    :param stem: Perform stemming
    :param remove_stopwords: Remove stopwords
    :param isolate_sym: Isolate symbols
    :param remove_alphanum: Remove non-alphanumeric
    :return: cleaned segments text

    """

    # Lower text
    if lower:
        text = text.lower()

    # Remove stopwords
    if remove_stopwords:
        pattern = re.compile(r"\b(" + r"|".join(stopwords.words('english')) + r")\b\s*")
        text = pattern.sub("", text)

    # Isolate symbols
    if isolate_sym:
        text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)

    # Remove non-alphanumeric
    if remove_alphanum:
        text = re.sub("[^A-Za-z0-9]+", " ", text)

    # Remove Extra padding
    text = re.sub(" +", " ", text)
    text = text.strip()

    # Remove hyperlinks
    text = re.sub(r"http\S+", "", text)

    # Perform stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word) for word in text.split(" ")])

    return text




def cleanHTML(html):

    html = html.split("</hr>")
    html = list(filter(None, html))
    h3_tags = re.compile('<h3>.*?</h3>')
    html = [re.sub(h3_tags, '', item) for item in html]
    tags = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    html = [re.sub(tags, '', item) for item in html]
    html = list(map(str.strip, html))
    html = list(filter(None, html))
    return html

class MyIter:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for fname in glob.glob(r"{}/*".format(self.path)):
            pol = os.path.join(fname, "dom_ind.html")
            processed_segments = getText(pol)
            for segment in processed_segments:
                yield word_tokenize(segment)


def getText(pol):
    try:
        html = open(pol, "r").read()
        list_of_segments = cleanHTML(html)
        #processed_segments = [preprocess.cleanText(segment, lower=params.lower, stem=params.stem) for segment in list_of_segments]
        processed_segments = [cleanText(segment, lower=True, stem=False) for segment in list_of_segments]
        return processed_segments
    except Exception as e:
        print(e)




if __name__ == '__main__':
    tic()
    data_path = "/Users/kaushik/Desktop/runasdus/src/com/lab/data/"
    model4 = FastText(vector_size=300,
                        window=10,
                        min_count=5,
                        sample=1e-3,
                        sg=1,
                        negative=5,
                        max_vocab_size=None,
                        workers=8)
    model4.build_vocab(corpus_iterable=MyIter(data_path))
    total_examples = model4.corpus_count
    model4.train(corpus_iterable=MyIter(data_path), total_examples=total_examples, epochs=5)

    toc()

    with tempfile.NamedTemporaryFile(prefix='saved_model_gensim-', delete=False) as tmp:
        model4.save(tmp.name, separately=[])

    # Load back the same model.
    loaded_model = FastText.load(tmp.name)
    print(loaded_model)

    os.unlink(tmp.name)
