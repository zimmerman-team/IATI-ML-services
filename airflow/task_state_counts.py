#!/usr/bin/env python3

import sqlalchemy as sa
from tabulate import tabulate

from common import config

def main():
    connstring = config.get_airflow_sqlalchemy_conn()
    engine = sa.create_engine(connstring)
    metadata = sa.MetaData()
    table = sa.Table("task_instance", metadata,
        sa.Column("task_id", sa.String,primary_key=True),
        sa.Column("state", sa.String),
        sa.Column("run_id", sa.String,primary_key=True),
        autoload_with=engine
    )
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
    conn = engine.connect()
    cursor = conn.execute(query)
    keys = cursor.keys()
    fetched = cursor.fetchall()
    data = [row.values() for row in fetched]
    buf = tabulate(data, headers=keys, tablefmt='psql')
    print(buf)

if __name__ == "__main__":
    main()