import numpy as np
import pandas as pd
import pickle
import urllib

from flask import render_template
from flask import request
import mlflow
import os
import re

from src.config import cfg
from src.data import prepOPPCorpus
from src.data import preprocess

from src.models import CNN
from src import models

from src.main import main
from src.main import driver

from src.flapp import app
from src.utils.inputparse import parseURL, text_process_policy, segmentParaRev, segmentPara
from src.utils import gen
from src.utils import metrics
from src.utils import embeddings

import itertools

experiment_name = "CNN_W_FE_M"
experiment_dpath = os.path.join(cfg.PARAM.BEST_PARAM_DPATH, "best_params_" + experiment_name)
run_id = gen.loadID(os.path.join(experiment_dpath, "run_ID.txt"))

#device = gen.setDevice(cuda=True)
#artifacts = driver.loadRunArtifacts(run_id=run_id, device = device)
#all_cats = artifacts['label_encoder'].classes
all_cats = ['Data Retention', 'Data Security', 'Do Not Track', 'First Party Collection/Use', 'International and Specific Audiences', 'Introductory/Generic', 'Policy Change', 'Practice not covered', 'Privacy contact information', 'Third Party Sharing/Collection', 'User Access, Edit and Deletion', 'User Choice/Control']
exclude_cats = ['Introductory/Generic', 'Practice not covered', 'Privacy contact information']
categories = [cat for cat in all_cats if cat not in exclude_cats]
cat_thresholds = {'Data Retention':1,
                 'Data Security':1,
                 'Do Not Track':1,
                 'First Party Collection/Use':5,
                 'International and Specific Audiences':1,
                 'Introductory/Generic':15,
                 'Policy Change':1,
                 'Practice not covered':30,
                 'Privacy contact information':15,
                 'Third Party Sharing/Collection':5,
                 'User Access, Edit and Deletion':1,
                 'User Choice/Control':1}

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/segclass')
def text_output():
    # Retrieve URL
    try:
        url = request.args.get('url_text')
        print('URL: ' + url)
    except:
        url = ''

    # Retrieve Text
    try:
        policy_text = request.args.get('policy_text')
        print('Policy text: ' + policy_text)
    except:
        policy_text = ''

    # INPUT error conditions
    domain = ''
    if url.strip() != '':
        # Try scraping the website
        try:
            text, domain = parseURL(url)

        except Exception as e:
            print(e)
            message = '<p>Unable to parse the URL.</p><p>Please check the URL and try again.</p>'
            return render_template("error.html", message=message)

        # Split the text into segments
        segment_list = segmentParaRev(text)

        # Error if no data scraped
        if len(segment_list) == 0:
            message = '<p>Unable to scrape textual data from the given URL.</p><p>Try providing the text directly.</p>'
            return render_template("error.html", message=message)

        # Error if less data scraped
        if len(' '.join(segment_list)) <= 500:
            message = '<p>The requested policy text is not big enough.</p><p>Please provide longer policies to obtain stable predictions.</p>'
            return render_template("error.html", message=message)

    elif policy_text.strip() != '':
        segment_list = segmentParaRev(policy_text)

        if len(' '.join(segment_list)) <= 500:
            message = '<p>The requested policy text is not big enough.</p><p>Please provide longer policies to obtain stable predictions.</p>'
            return render_template("error.html", message=message)

    else:
        message = '<p>No input provided.</p><p>Please provide a <strong>URL</strong> or <strong>TEXT</strong> data of an individual privacy policy.</p>'
        return render_template("error.html", message=message)  # send 'em back home

    # Raw segments
    orig_segments = pd.DataFrame({'segments': segment_list})
    segments_processed = [text_process_policy(segment) for segment in segment_list]
    regex = re.compile(r"^[^A-Za-z0-9]+")
    segments_processed = [regex.sub("", segment) for segment in segments_processed]
    segments_processed = [re.sub(" +", " ", segment).strip() for segment in segments_processed]
    segments_processed = [segment for segment in segments_processed if segment.strip() != '']   # Remove blank lines
    segments_processed = [segment for segment in segments_processed if len(segment.split()) > 1]
    segments_processed_df = pd.DataFrame({'segments': segments_processed})

    #Package segments info to Streamlit UI
    segments_pkg = gen.packageSegments(segments_processed)
    gen.savePickle(segments_pkg, "/home/user/appdata/segments.pkl")
    gen.savePickle(domain, "/home/user/appdata/domain.pkl")
    gen.savePickle(url, "/home/user/appdata/url.pkl")

    # Get predictions for segments
    confidence_segments, tagged_segments = driver.productionPredict(segments_processed, run_id, multi_threshold = True)
    # Get total category counts for all segments
    results = tagged_segments.sum()

    # Merge original segments with the predictions
    tagged_segments = pd.concat([tagged_segments, segments_processed_df], axis=1)
    cat_content = {}
    segments = {}
    trigger = {}
    confidence = {}
    sitelists = {}


    for cat in categories:
        confidence[cat], segments[cat] = gen.rankSegments(list(confidence_segments[tagged_segments[cat] == 1][cat]), list(tagged_segments[tagged_segments[cat] == 1]['segments']))
        # Trigger flags if greater than thresholds
        trigger[cat] = results[cat] >= cat_thresholds[cat]
        sitelists[cat] = gen.sitelistGen(segments[cat], url, num_links = 3)
        assert(len(confidence[cat]) == len(segments[cat]) == len(sitelists[cat]))

    cat_content['segments'] = segments
    cat_content['trigger'] = trigger
    cat_content['confidence'] = confidence
    cat_content['sitelists'] = sitelists


    #DEBUGGING
    #gen.savePickle(segments, "segments.pkl")
    #gen.savePickle(trigger, "trigger.pkl")
    #gen.savePickle(confidence, "confidence.pkl")
    #gen.savePickle(sitelists, "sitelists.pkl")
    #gen.savePickle(url, "url.pkl")

    return render_template("polspec.html", policy_text=policy_text,
                           cat_content = cat_content,
                           domain=domain,
                           url=url)

