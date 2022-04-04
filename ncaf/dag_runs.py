#!/usr/bin/env python3

import sqlalchemy as sa

from common import config
from . import sa_common
from . import sa_tables

table = sa_tables.dag_run
query = sa.select([
    table.columns.dag_id,
    table.columns.run_id
]).order_by(
    table.columns.start_date.desc()
)


def main():
    keys,data = sa_common.fetch_data(query)
    sa_common.print_data(keys,data)

if __name__ == "__main__":
    main()
