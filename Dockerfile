FROM python:3.7.4-stretch

RUN mkdir -p /home/user/app
WORKDIR /home/user/app

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# install as a package
COPY requirements.txt /home/user/app
RUN pip install --no-cache-dir -r requirements.txt

# copy data
COPY OPP-115 /home/user/app/OPP-115

# copy code
COPY src /home/user/app/src
COPY mlflow_registry /home/user/app/mlflow_registry
COPY gunicorn.py /home/user/app/gunicorn.py

#Change Dir permissions
RUN chmod -R 777 /home/user/app

# Application Environment variables
#ENV APP_ENV development
ENV PORT 8777

# Exposing Ports
EXPOSE $PORT

# cmd for running the API
CMD gunicorn -b :$PORT -c gunicorn.py src.run:app
