DROP DATABASE airflow_db;
DROP USER airflow_user;
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'AIRFLOW_PG_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- command-line access to the db: # psql -d airflow_db -h localhost -U airflow_user -W