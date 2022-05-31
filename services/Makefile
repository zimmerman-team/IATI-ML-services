all:
	echo "targets: up, build, down"

up: .env
	sudo docker image prune -f
	sudo usermod -a -G 999 ${USER}
	sudo chown 999:999 -Rf mongo/db
	sudo chown 1000:1000 -Rf airflow_scheduler/trained_models
	sudo chown 1000:1000 -Rf airflow_scheduler/tmp_large_mp
	sudo chown 10001:10001 -Rf loki/wal
	sudo chmod g+rw -Rf mongo/db
	find airflow_scheduler/tmp_large_mp -type s,p | while read X;  do rm -v "$X" ; done
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up -d
	bash cadvisor/run.sh

.env:
	cd .. && python3 -c "from common import config; config.print_env() "> ./services/.env

build:
	sudo docker image prune -f
	bash generate_airflow_cfg.sh airflow_scheduler/airflow.cfg
	bash generate_airflow_cfg.sh airflow_webserver/airflow.cfg
	# the directory code will be added in the container as it has the dags
	rm -rvf /tmp/airflow_scheduler_code_old/
	mv -vf ./airflow_scheduler/code /tmp/airflow_scheduler_code_old
	mkdir -pv ./airflow_scheduler/code
	find airflow_scheduler/tmp_large_mp/ -type s,p | while read X;  do rm -v "$X" ; done
	cp -r ../ncaf ../model_config ../dspn_annotated ../multiset-equivariance ../common ../dags ../preprocess ../models ./airflow_scheduler/code
	bash complete_airflow_pg_db.sh
	COMPOSE_PROJECT_NAME=learning_sets docker-compose build
	sudo docker image prune -f

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
