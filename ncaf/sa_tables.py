import sqlalchemy as sa
from . import sa_common

print("defining task_instance..")
task_instance = sa.Table(
    "task_instance",
    sa_common.metadata,
    sa.Column("task_id", sa.String, primary_key=True),
    sa.Column("state", sa.String),
    sa.Column("run_id", sa.String, primary_key=True),
    sa.Column('start_date',sa.DateTime),
    sa.Column('end_date', sa.DateTime),
    extend_existing=True,
    autoload_with=sa_common.engine
 )

print("done. defining dag_run..")
dag_run = sa.Table(
    "dag_run",
    sa_common.metadata,
    sa.Column("dag_id", sa.String),
    sa.Column("run_id", sa.String, primary_key=True),
    sa.Column("start_date", sa.DateTime),
    sa.Column("execution_date", sa.DateTime),
    sa.Column("state", sa.String),
    extend_existing=True,
    autoload_with=sa_common.engine
)
print("done!")