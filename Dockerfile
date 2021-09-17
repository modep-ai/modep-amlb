FROM ubuntu:18.04

WORKDIR /bench

ENV DEBIAN_FRONTEND noninteractive

ARG ssh_prv_key
ARG ssh_pub_key
ARG gcp_creds

RUN apt-get update && \
    apt-get install -y apt-utils dialog locales curl wget unzip git emacs-nox software-properties-common openssh-server libpq-dev redis-server supervisor

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions using keys based as build args
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

# Create GCP credentials file using base64 encoded build arg
ENV GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/gcp-creds.json
RUN mkdir -p /var/secrets
RUN echo "$gcp_creds" > /var/secrets/gcp-creds-base64.json
RUN base64 -d /var/secrets/gcp-creds-base64.json > $GOOGLE_APPLICATION_CREDENTIALS

# install python
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.7 python3.7-venv python3.7-dev python3-pip
# RUN update-alternatives --install /usr/bin/python3 python3 $(which python3.7) 1

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

# ADD modep_amlb /bench/modep_amlb
ADD . /bench/modep_amlb

# ADD automlbenchmark /bench/automlbenchmark
RUN git clone git@github.com:jimgoo/automlbenchmark.git /bench/automlbenchmark

# ADD app_utils /bench/app_utils
RUN git clone git@github.com:jimgoo/modep-common.git /bench/modep-common

# remove installation artifacts
RUN rm -rf /bench/automlbenchmark/frameworks/*/venv
RUN rm -rf /bench/automlbenchmark/frameworks/*/.installed
RUN rm -rf /bench/automlbenchmark/frameworks/*/.setup_env
RUN rm -rf /bench/automlbenchmark/frameworks/*/lib

# install common
RUN $PIP install --no-cache-dir -r /bench/modep-common/requirements.txt
RUN $PIP install --no-cache-dir /bench/modep-common/

# install modep_amlb
RUN $PIP install --no-cache-dir -r /bench/modep_amlb/requirements.txt
# -e needed for templates to be in the right place
RUN $PIP install --no-cache-dir /bench/modep_amlb/

# install automlbenchmark in the same order as requirements.txt
RUN (grep -v '^\s*#' | xargs -L 1 $PIP install --no-cache-dir) < /bench/automlbenchmark/requirements.txt

# run setup for a single test framework
RUN $PY /bench/automlbenchmark/runbenchmark.py constantpredictor -s only

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
# RUN $PY /bench/automlbenchmark/runbenchmark.py constantpredictor -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py randomforest -s only
# RUN $PY /bench/automlbenchmark/runbenchmark.py tunedrandomforest -s only

EXPOSE 8080

CMD ["supervisord"]
