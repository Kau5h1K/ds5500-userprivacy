from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib
from nltk.stem.snowball import *
import re
import string
from src.utils import gen

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def getTextFromHTML(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u"\n".join(t.strip() for t in visible_texts)


def parseURL(url):
    page = urllib.request.urlopen(url)

    # Check we got a good response, otherwise complain
    if str(page.code)[0] != '2':
        e = urllib.error.URLError('The server returned status code ' + str(page.code))
        raise e

    content = page.read()

    # Parse URL paragraphs
    pars = getTextFromHTML(content)

    # Parse domain
    domain = '.'.join(url.split('//')[-1].split('/')[0].split('.')[-2:])
    return pars, domain


def text_process_policy(doc):

    s = re.sub(r'\<.*\>','',doc)

    return s

def segmentParaRev(doc):
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


def segmentPara(doc):

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


if __name__ == '__main__':
    segments = gen.loadPickle("segments.pkl")
    trigger = gen.loadPickle("trigger.pkl")
    confidence = gen.loadPickle("confidence.pkl")
    sitelists = gen.loadPickle("sitelists.pkl")
    url = gen.loadPickle("url.pkl")
    #sitelistGen(segments['User Choice/Control'], url, num_links = 3)