import sqlalchemy as sa
from tabulate import tabulate

from common import config

connstring = config.get_airflow_sqlalchemy_conn()
engine = sa.create_engine(connstring)
metadata = sa.MetaData()

def fetch_data(query):
    with engine.connect() as conn:
        cursor =  conn.execute(query)
        keys = cursor.keys()
        fetched = cursor.fetchall()
        data = [row.values() for row in fetched]
        return keys,data

def print_data(keys,data):
    buf = tabulate(
        data,
        headers=keys,
        tablefmt='psql'
    )
    print(buf)
