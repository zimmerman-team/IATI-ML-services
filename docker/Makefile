all:
	echo "targets: up, build, down"

up: .env
	COMPOSE_PROJECT_NAME=learning_sets docker-compose up

.env:
	cd .. && python3 -c "from common import config; config.print_env() "> docker/.env

build:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose build

down:
	COMPOSE_PROJECT_NAME=learning_sets docker-compose down
