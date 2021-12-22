import os
import sys

import logging
import pandas as pd
from json import JSONDecodeError
from pathlib import Path
import streamlit as st
from annotated_text import annotation
from markdown import markdown
import pickle

from haystack.nodes import FARMReader, TransformersReader
from haystack.utils import launch_es
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever
from haystack.pipelines import ExtractiveQAPipeline



# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import query, get_backlink


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "How do you collect my data?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "Website")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", 10))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", 3))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", Path(__file__).parent / "random_questions.csv")
FAVICON = os.getenv("FAVICON_FILE", Path(__file__).parent / "favicon.png")

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD", 1))


try:
    launch_es()
except:
    pass

document_store = ElasticsearchDocumentStore(host="elasticsearch", port=9200, username="", password="", index="document")

document_store.delete_documents(filters={})

try:
    with open("/home/user/appdata/segments.pkl", "rb") as f:
        segments_dict = pickle.load(f)
    with open("/home/user/appdata/domain.pkl", "rb") as f:
        domain = pickle.load(f)
    with open("/home/user/appdata/url.pkl", "rb") as f:
        url_pol = pickle.load(f)
except:
    reload_mode = 1


document_store.write_documents(segments_dict)
retriever = ElasticsearchRetriever(document_store=document_store)



def main():
    reload_mode = 0
    #resp_bool = reset_datastore()

    st.set_page_config(page_title='Privacy Policy QA', page_icon=FAVICON)
    # if debug:
    #
    #     if resp_bool:
    #         st.info('Deleted all documents')
    #     else:
    #         st.info('Failed to delete all documents')

    #index_datastore(segments_dict)

    # Persistent state
    state = SessionState.get(
        question=DEFAULT_QUESTION_AT_STARTUP,
        answer=DEFAULT_ANSWER_AT_STARTUP,
        results=None,
        raw_json=None,
        random_question_requested=False
    )

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        state.answer = None
        state.results = None
        state.raw_json = None

    # Title
    st.write("# Privacy Policy Question Answering")
    if reload_mode:
        st.error("Privacy Policy state information not captured properly. Please reload the page!")
    if len(domain) and len(url_pol):
        st.markdown("""
        <h3 style='text-align:center;padding: 0 0 1rem;'>Ask Me Anything about <a href="{1}">{0}</a>!</h3>
        
        Ask any question related to privacy practices carried out by {0} that you'd like to know about! Try clicking on Random question to see a sample query.
        
        *Note: do not use keywords, but full-fledged questions.* The underlying models are not optimized to deal with keyword queries and might misunderstand you.
        """.format(domain.capitalize(), url_pol), unsafe_allow_html=True)
    else:
        st.markdown("""
        <h3 style='text-align:center;padding: 0 0 1rem;'>Ask Me Anything!</h3>
        
        Ask any question related to privacy practices carried out by this company that you'd like to know about! Try clicking on Random question to see a sample query.
        
        *Note: do not use keywords, but full-fledged questions.* The underlying models are not optimized to deal with keyword queries and might misunderstand you.
        """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Options")
    top_k_reader = st.sidebar.slider(
        "Max. number of answers",
        min_value=1,
        max_value=10,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        on_change=reset_results)
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=len(segments_dict),
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results)
    #debug = st.sidebar.checkbox("Show debug info")
    model_name = st.sidebar.selectbox("Select one of READER models", (
        "roberta-base-squadqa", "robertabase_squadqa_policyqa", "privbert_squadqa", "privbert_squadqa_policyqa", "bertbase_squadqa_policyqa"))

    if model_name == "roberta-base-squadqa":
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    else:
        reader = TransformersReader(model_name_or_path="./QA_models/{}".format(model_name), use_gpu=False)

    pipe = ExtractiveQAPipeline(reader, retriever)

    st.sidebar.markdown(f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .haystack-footer {{
            text-align: center;
        }}
        .haystack-footer h4 {{
            margin: 0.1rem;
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    <div class="haystack-footer">
        <hr />
        <small>Detecting Textual Saliency in Privacy Policy.</small>
    </div>
    """, unsafe_allow_html=True)

    # Load csv into pandas dataframe
    try:
        df = pd.read_csv(EVAL_LABELS, sep=";")
    except Exception:
        st.error(f"The eval file was not found. Please check the demo's [README](https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")
        sys.exit(f"The eval file was not found under `{EVAL_LABELS}`. Please check the README (https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")

    # Search bar
    question = st.text_input("",
        value=state.question,
        max_chars=100,
        on_change=reset_results
    )
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    # Get next random question from the CSV
    if col2.button("Random question"):
        reset_results()
        new_row = df.sample(1)
        while new_row["Question Text"].values[0] == state.question:  # Avoid picking the same question twice (the change is not visible on the UI)
            new_row = df.sample(1)
        state.question = new_row["Question Text"].values[0]
        state.answer = new_row["Answer"].values[0]
        state.random_question_requested = True
        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
    else:
        state.random_question_requested = False
    
    run_query = (run_pressed or question != state.question) and not state.random_question_requested

    # Get results for query
    if run_query and question:
        reset_results()
        state.question = question
        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on the privacy policy document..."
        ):
            try:
                state.results, state.raw_json = query(pipe, question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Cannot access the document store!")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    if state.results:

        st.write("## Results:")

        for count, result in enumerate(state.results):
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                st.write(markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#65AAC3")) + context[end_idx:]), unsafe_allow_html=True)
                source = ""
                url, title = get_backlink(result)
                if url and title:
                    source = f"[{result['document']['meta']['title']}]({result['document']['meta']['url']})"
                else:
                    source = f"{result['source']}"
                st.markdown(f"**Relevance:** {result['relevance']} -  **Source:** {source}")

            else:
                st.write("🤔 &nbsp;&nbsp; We are unsure whether the policy document contains an answer to your question. Try to reformulate it!")
                st.write("**Relevance:** ", result["relevance"])

            st.write("___")

        # if debug:
        #     st.subheader("REST API JSON response")
        #     st.write(state.raw_json)

main()
