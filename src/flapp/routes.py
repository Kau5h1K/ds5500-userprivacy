import numpy as np
import pandas as pd
import pickle
import urllib

from flask import render_template
from flask import request
import mlflow
import os

from src.config import cfg
from src.data import prepOPPCorpus
from src.data import preprocess

from src.models import CNN
from src import models

from src.main import main
from src.main import driver

from src.flapp import app
from src.utils.inputparse import url_input_parser, text_process_policy, reverse_paragraph_segmenter, post_process_segments
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
            text, domain = url_input_parser(url)

        except Exception as e:
            print(e)
            message = '<p>Unable to parse the URL.</p><p>Please check the URL and try again.</p>'
            return render_template("error.html", message=message)

        # Split the text into segments
        segment_list = reverse_paragraph_segmenter(text)

        # Error if no data scraped
        if len(segment_list) == 0:
            message = '<p>Unable to scrape textual data from the given URL.</p><p>Try providing the text directly.</p>'
            return render_template("error.html", message=message)

        # Error if less data scraped
        if len(' '.join(segment_list)) <= 500:
            message = '<p>The requested policy text is not big enough.</p><p>Please provide longer policies to obtain stable predictions.</p>'
            return render_template("error.html", message=message)

    elif policy_text.strip() != '':
        segment_list = reverse_paragraph_segmenter(policy_text)

        if len(' '.join(segment_list)) <= 500:
            message = '<p>The requested policy text is not big enough.</p><p>Please provide longer policies to obtain stable predictions.</p>'
            return render_template("error.html", message=message)

    else:
        message = '<p>No input provided.</p><p>Please provide a <strong>URL</strong> or <strong>TEXT</strong> data of an individual privacy policy.</p>'
        return render_template("error.html", message=message)  # send 'em back home

    # Raw segments
    orig_segments = pd.DataFrame({'segments': segment_list})
    segments_processed = [text_process_policy(segment) for segment in segment_list]
    segments_processed = [segment for segment in segments_processed if segment.strip() != '']   # Remove blank lines

    # Get predictions for segments
    tagged_segments = driver.productionPredict(segments_processed, run_id, multi_threshold = True)

    # Get total category counts for all segments
    results = tagged_segments.sum()

    # Merge original segments with the predictions
    tagged_segments = pd.concat([tagged_segments, orig_segments], axis=1)
    print(list(tagged_segments[tagged_segments['Policy Change'] == 1]['segments']))
    segments = {}
    trigger = {}
    for cat in categories:
        segments[cat] = list(tagged_segments[tagged_segments[cat] == 1]['segments'])
        segments[cat] = post_process_segments(segments[cat])

        # Trigger flags if greater than thresholds
        trigger[cat] = results[cat] >= cat_thresholds[cat]

    return render_template("polspec.html", policy_text=policy_text,
                           segments_data_security=segments['Data Security'],
                           segments_data_retention=segments['Data Retention'],
                           segments_do_not_track=segments['Do Not Track'],
                           segments_first_party_collection=segments['First Party Collection/Use'],
                           segments_third_party_sharing=segments['Third Party Sharing/Collection'],
                           segments_user_access=segments['User Access, Edit and Deletion'],
                           segments_policy_change=segments['Policy Change'],
                           segments_user_choice=segments['User Choice/Control'],
                           segments_spl_audience=segments['International and Specific Audiences'],
                           bool_data_security=trigger['Data Security'],
                           bool_data_retention=trigger['Data Retention'],
                           bool_do_not_track=trigger['Do Not Track'],
                           bool_first_party_collection=trigger['First Party Collection/Use'],
                           bool_third_party_sharing=trigger['Third Party Sharing/Collection'],
                           bool_user_access=trigger['User Access, Edit and Deletion'],
                           bool_policy_change=trigger['Policy Change'],
                           bool_user_choice=trigger['User Choice/Control'],
                           bool_spl_audience=trigger['International and Specific Audiences'],
                           domain=domain,
                           url=url)

