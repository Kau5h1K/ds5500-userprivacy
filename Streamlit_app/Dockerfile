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

#Download and copy trained models
RUN gdown --id 1JKjWvOVs1x4zT0PyOF1zFZDp78xnCqD_
COPY pytorch_model.bin /home/user/app_qa/QA_models/bertbase_squadqa_policyqa/

RUN gdown --id 1eHZOphDTy6cAwyx9UDbJyItsMXqjbx4X
COPY pytorch_model.bin /home/user/app_qa/QA_models/privbert_squadqa/

RUN gdown --id 1EV0X1E9-Akk_TE6la0aiPPBDmBI6L2Vv
COPY pytorch_model.bin /home/user/app_qa/QA_models/privbert_squadqa_policyqa/

RUN gdown --id 1P9tgeGEekU7xgtjRgDWU0VXkyuGBZyoe
COPY pytorch_model.bin /home/user/app_qa/QA_models/robertabase_squadqa_policyqa/

EXPOSE 8501

# cmd for running the API
CMD ["streamlit", "run", "AskMeAnything.py"]
