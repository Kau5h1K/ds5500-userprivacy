from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib
from nltk.stem.snowball import *
import re
import string


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u"\n".join(t.strip() for t in visible_texts)


def url_input_parser(url):
    """
    Input is url as a string. Parses body text and returns
    string of all paragraph elements in doc.
    Secondarily, searches the input url for its domain and returns
    domain in the form 'google.com'.
    """

    # Request URL
    page = urllib.request.urlopen(url)

    # Check we got a good response, otherwise complain
    if str(page.code)[0] != '2':
        e = urllib.error.URLError('The server returned status code ' + str(page.code))
        raise e

    content = page.read()

    # Parse URL paragraphs
    pars = text_from_html(content)

    # Parse domain
    domain = '.'.join(url.split('//')[-1].split('/')[0].split('.')[-2:])
    return pars, domain



def text_process_old(doc):
    """
    1. remove punctuation
    2. remove stopwords
    3. remove HTML tags
    4. remove '|||' inserted into corpus documents only
    """
    lst = [word for word in doc.split() if re.search(r'\<.*\>', word) is None]
    lst = ' '.join(lst)
    lst = [char for char in lst if char not in string.punctuation]
    lst = ''.join(lst)
    lst = [word for word in lst.split() if word.lower() not in stopwords.words('english')]
    lst = [word for word in lst if word.replace('|||', '')]
    return ' '.join(lst)


def text_process_policy(doc):
    """
    Takes in doc as string and returns a processed string
    by performing the following steps:
    1. remove HTML tags
    2. remove punctuation
    3. remove stopwords
    4. stemming
    5. remove any blank lines
    """

    s = re.sub(r'\<.*\>','',doc)

    return s

def reverse_paragraph_segmenter(doc):
    """
    input: doc as string
    output: list of paragraphs with blank lines removed, concatenated
    in reverse to that topic headings are concatenated with the
    paragraphs below them.
    """

    lines = doc.split('\n')
    segs = list()
    segs.append(lines[-1])
    c = -1
    for line in lines[::-1]:
        if len(line) < 75:  # Less than 75 chars wide in line join to prev line
            segs[c] = ' '.join([line, segs[c]])
        else:
            segs.insert(0, line)
            c -= 1

    return [line for line in segs if line.strip() != '']


def text_paragraph_segmenter(doc):
    """
    input: doc as string
    output: list of paragraphs with blank lines removed
    """

    lines = doc.split('\n')
    segs = list()
    segs.append(lines[0])
    c = 0
    for i in range(1, len(lines)):
        if len(lines[i]) < 75:  # Less than 75 chars wide in line join to prev line
            segs[c] = ' '.join([segs[c], lines[i]])
        else:
            segs.append(lines[i])
            c += 1

    return [line for line in segs if line.strip() != '']
