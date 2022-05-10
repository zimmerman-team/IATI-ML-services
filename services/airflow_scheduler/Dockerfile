FROM bitnami/airflow-scheduler:latest

USER root
RUN apt update
RUN apt install -y iputils-ping postgresql-common libpq-dev gcc

RUN . /opt/bitnami/airflow/venv/bin/activate && pip install cython torchvision ipython hiddenlayer matplotlib seaborn apache-airflow apache-airflow[postgres] psycopg2 numpy sklearn pymongo torch mlflow gensim nltk pytorch_lightning ray wandb higher billiard statsd npyscreen
RUN . /opt/bitnami/airflow/venv/bin/activate && pip install scikit-learn mlflow
RUN . /opt/bitnami/airflow/venv/bin/activate && pip install pyyaml git+https://github.com/BenjaminDoran/unidip
RUN . /opt/bitnami/airflow/venv/bin/activate && python3 -c "import nltk; nltk.download('punkt')"
RUN . /opt/bitnami/airflow/venv/bin/activate && pip install torch

ADD airflow.cfg /opt/bitnami/airflow
ADD add_dag_bags.py /opt/bitnami/airflow/dags/

# changing some permission to the 'airflow' user (UID 1000)
RUN chown 1000:1000 /opt/bitnami/airflow -R

# the directory code is created by make build
USER 1000
ADD code /learning_sets/
RUN touch /learning_sets/dspn_annotated/dspn/__init__.py

USER root
RUN mkdir -p /learning_sets/config/

USER 1000
ADD airflow_scheduler.yaml /learning_sets/config/

USER root
RUN chown 1000:1000 /learning_sets/ -R

USER 1000
# required to correctly load the dags
#ENTRYPOINT 'bash /opt/bitnami/scripts/airflow-scheduler/run.sh'
