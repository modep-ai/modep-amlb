FROM ubuntu:18.04

WORKDIR /bench
ENV DEBIAN_FRONTEND noninteractive
ARG gcp_creds

RUN apt-get update && \
    apt-get install -y apt-utils dialog locales curl wget unzip git \
    emacs-nox software-properties-common htop \
    openssh-server libpq-dev redis-server supervisor

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Create GCP credentials file using base64 encoded build arg
ENV GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/gcp-creds.json
RUN mkdir -p /var/secrets
RUN echo "$gcp_creds" > /var/secrets/gcp-creds-base64.json
RUN base64 -d /var/secrets/gcp-creds-base64.json > $GOOGLE_APPLICATION_CREDENTIALS

# install python
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.7 python3.7-venv python3.7-dev python3-pip

# aliases for the python system
ENV SPIP python3.7 -m pip
ENV SPY python3.7

# Enforce UTF-8 encoding
ENV PYTHONUTF8 1
ENV PYTHONIOENCODING utf-8
# RUN locale-gen en-US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
RUN $SPIP install -U pip wheel
RUN $SPY -m venv venv
ENV PIP /bench/venv/bin/python3 -m pip
ENV PY /bench/venv/bin/python3 -W ignore
# RUN $PIP install -U pip==None wheel
RUN $PIP install -U pip wheel

VOLUME /input
VOLUME /output
VOLUME /custom

# setup supervisord and redis
RUN mkdir -p /etc/supervisor/conf.d/
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY ./redis.conf /etc/redis.conf

# invalidate the docker cache for these repos so that git clone always clones the latest
ADD https://api.github.com/repos/modep-ai/modep-common/git/refs/heads/main /bench/modep-common-version.json
ADD https://api.github.com/repos/modep-ai/automlbenchmark/git/refs/heads/master /bench/automlbenchmark-version.json

# get all code
RUN git clone https://github.com/modep-ai/modep-common.git /bench/modep-common
RUN git clone https://github.com/modep-ai/automlbenchmark.git /bench/automlbenchmark
ADD . /bench/modep-amlb

# remove installation artifacts just in case
RUN rm -rf /bench/automlbenchmark/frameworks/*/venv
RUN rm -rf /bench/automlbenchmark/frameworks/*/.installed
RUN rm -rf /bench/automlbenchmark/frameworks/*/.setup_env
RUN rm -rf /bench/automlbenchmark/frameworks/*/lib

# install modep-common
RUN $PIP install --no-cache-dir -r /bench/modep-common/requirements.txt
RUN $PIP install --no-cache-dir /bench/modep-common/

# install modep-amlb (-e needed for templates to be in the right place)
RUN $PIP install --no-cache-dir -r /bench/modep-amlb/requirements.txt
RUN $PIP install --no-cache-dir /bench/modep-amlb/

# install automlbenchmark in the same order as requirements.txt
RUN (grep -v '^\s*#' | xargs -L 1 $PIP install --no-cache-dir) < /bench/automlbenchmark/requirements.txt

## run setup for all frameworks that we want to use
# RUN $PY /bench/automlbenchmark/runbenchmark.py autogluon -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py autosklearn -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py autoweka -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py flaml -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py gama -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py h2oautoml -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py hyperoptsklearn -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py mljarsupervised -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py mlnet -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py tpot -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py constantpredictor -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py randomforest -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py tunedrandomforest -s only

EXPOSE 8080

# CMD ["supervisord"]
# ENTRYPOINT ["/bench/venv/bin/python3", "/bench/modep-amlb/modep_amlb/cli.py"]

