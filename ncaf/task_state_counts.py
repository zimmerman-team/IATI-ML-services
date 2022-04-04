#!/usr/bin/env python3

import sqlalchemy as sa
from tabulate import tabulate

from common import config
from . import sa_common
from . import sa_tables

table = sa_tables.task_instance

def query_by_run_id(run_id):
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