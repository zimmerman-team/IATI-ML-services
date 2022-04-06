#!/usr/bin/env python3

import sqlalchemy as sa
from tabulate import tabulate

from common import config
from . import sa_common
from . import sa_tables

table = sa_tables.task_instance

def task_instances_by_state(run_id, state):
    query = sa.select([
        table.columns.task_id,
        table.columns.start_date,
        table.columns.end_date
    ]).order_by(
        table.columns.end_date.desc()
    ).where(
        table.columns.run_id == run_id
    ).where(
        table.columns.state == state
    )
    return query

def task_state_counts_by_run_id(run_id):
    query = sa.select([
        sa.func.count(table.columns.state).label('count'),
        table.columns.state,
        table.columns.run_id
    ]).group_by(
        table.columns.state,
        table.columns.run_id
    ).order_by(
        table.columns.run_id.desc()
    ).where(
        table.columns.run_id == run_id
    )
    return query

def dag_runs():
    table = sa_tables.dag_run
    query = sa.select([
        table.columns.dag_id,
        table.columns.state,
        table.columns.run_id,
        table.columns.start_date
    ]).order_by(
        table.columns.start_date.desc()
    )
    return query

#def main():
#    keys,data = sa_common.fetch_data(query)
#    sa_common.print_data(keys,data)

def main():

    query = sa.select([
        sa.func.count(table.columns.state).label('count'),
        table.columns.state,
        table.columns.run_id
    ]).group_by(
        table.columns.state,
        table.columns.run_id
    ).order_by(
        table.columns.run_id.desc()
    )
    conn = sa_common.engine.connect()
    cursor = conn.execute(query)
    keys = cursor.keys()
    fetched = cursor.fetchall()
    data = [row.values() for row in fetched]
    buf = tabulate(data, headers=keys, tablefmt='psql')
    print(buf)

if __name__ == "__main__":
    main()