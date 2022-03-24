#!/bin/bash

# exit when any command fails
#set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

apt install -y m4 postgresql python3-pip libpq-dev gnupg

# FIXME: to requirement.txt
pip install cython torchvision ipython hiddenlayer matplotlib seaborn apache-airflow apache-airflow[postgres] psycopg2 numpy sklearn pymongo torch mlflow gensim nltk pytorch_lightning ray wandb higher billiard statsd
pip install git+https://github.com/BenjaminDoran/unidip
python3 -c "import nltk; nltk.download('punkt')"

wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
#UBUNTU_CODENAME=$(lsb_release -dc | grep Codename | awk '{print $2}')
UBUNTU_CODENAME=focal
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu $UBUNTU_CODENAME/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
apt update
sudo apt-get install -y mongodb-org
service mongod start

MONGO_USER="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh mongo_user)"
MONGO_PASSWORD="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh mongo_password)"

mongosh <<- EOM
use learning_sets;
db.dropUser("$MONGO_USER");
db.createUser({ user: "$MONGO_USER", pwd: "$MONGO_PASSWORD", roles: [ { role: "readWrite", db: "soccer" }] });

EOM

# FIXME: automate conf item extraction?
AIRFLOW_PG_PASSWORD="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_pg_password)"
AIRFLOW_USER="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_user)"
AIRFLOW_PASSWORD="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_password)"
AIRFLOW_EMAIL="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_email)"
AIRFLOW_CONCURRENCY="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_concurrency)"

# exit 0 here prevents a failing killall to interrupt the script
( killall airflow ; exit 0 )

m4 -DAIRFLOW_PG_PASSWORD=$AIRFLOW_PG_PASSWORD psql_commands.m4 |
  sudo su postgres -c "/usr/bin/psql"

mkdir -pv $HOME/airflow
mkdir -pv $HOME/airflow/dags

# will macroexpand HOME with the user's home directory
m4 -DHOME=$HOME \
   -DAIRFLOW_PG_PASSWORD=$AIRFLOW_PG_PASSWORD \
   -DAIRFLOW_CONCURRENCY=$AIRFLOW_CONCURRENCY \
   airflow.cfg.m4 > $HOME/airflow/airflow.cfg

# will set the learning_sets dir (extracted from this script's path) to the module that will add learning_sets' dag
m4 -DLEARNING_SETS_DIR=$LEARNING_SETS_DIR \
   add_dag_bags.py.m4 > $HOME/airflow/dags/add_dag_bags.py

airflow db init

airflow users create -u $AIRFLOW_USER -e "$AIRFLOW_EMAIL" -r Admin -f $AIRFLOW_USER -l X -p "$AIRFLOW_PASSWORD"

airflow pools set npas_intensive 1 "creation of large npas may require a lot of resources"

pip3 install 'apache-airflow[statsd]'
