all:
	echo "targets: up, build, down"

up: .env
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d

.env:
	cd .. && python3 -c "from common import config; config.print_env() "> ./services/.env

build:
	bash generate_airflow_cfg.sh airflow_scheduler/airflow.cfg
	bash generate_airflow_cfg.sh airflow_webserver/airflow.cfg
	COMPOSE_PROJECT_NAME=learning_sets docker-compose build

down:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose down

status:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose ps

airflow_up:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d postgresql_for_airflow
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d airflow_webserver
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d airflow_scheduler

