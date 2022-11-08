from airflow import settings
from airflow.models import TaskInstance
from airflow.utils.state import State
import logging

def print_running_tasks():
    session = settings.Session()
    for task in session.query(TaskInstance) \
            .filter(TaskInstance.state == State.RUNNING) \
            .all():
        logging.info(f'task_id: {task.task_id}, dag_id: {task.dag_id}, start_date: {task.start_date}, '
                     f'hostname: {task.hostname}, unixname: {task.unixname}, job_id: {task.job_id}, pid: {task.pid}')

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    print_running_tasks()

