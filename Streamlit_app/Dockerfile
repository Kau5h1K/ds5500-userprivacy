FROM python:3.7.4-stretch

RUN mkdir -p /home/user/app_qa
WORKDIR /home/user/app_qa

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# install as a package
COPY requirements.txt /home/user/app_qa/
RUN pip install -r requirements.txt

# copy code
COPY utils.py /home/user/app_qa/
COPY AskMeAnything.py /home/user/app_qa/
COPY random_questions.csv /home/user/app_qa/
COPY SessionState.py /home/user/app_qa/
COPY favicon.png /home/user/app_qa/
COPY QA_models /home/user/app_qa/QA_models


EXPOSE 8501

# cmd for running the API
CMD ["streamlit", "run", "AskMeAnything.py"]
