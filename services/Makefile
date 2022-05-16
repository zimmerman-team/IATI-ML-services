all:
	echo "targets: up, build, down"

up: .env
	sudo usermod -a -G 999 ${USER}
	sudo chown 999:999 -Rf mongo/db
	sudo chmod g+rw -Rf mongo/db
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d

.env:
	cd .. && python3 -c "from common import config; config.print_env() "> ./services/.env

build:
	bash generate_airflow_cfg.sh airflow_scheduler/airflow.cfg
	bash generate_airflow_cfg.sh airflow_webserver/airflow.cfg
	# the directory code will be added in the container as it has the dags
	rm -rvf /tmp/airflow_scheduler_code_old/
	mv -vf ./airflow_scheduler/code /tmp/airflow_scheduler_code_old
	mkdir -pv ./airflow_scheduler/code
	cp -r ../model_config ../dspn_annotated ../multiset-equivariance ../common ../dags ../preprocess ../models ./airflow_scheduler/code
	COMPOSE_PROJECT_NAME=learning_sets docker-compose build

down:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose down

status:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose ps

airflow_up:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d postgresql_for_airflow
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d airflow_webserver
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d airflow_scheduler

debug_airflow_scheduler:
	make down
	make build
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d postgresql_for_airflow
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d airflow_webserver
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up airflow_scheduler
