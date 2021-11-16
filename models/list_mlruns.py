import mlflow
import argparse
import sys
import os
import collections
from dataclasses import dataclass

from tabulate import tabulate

sys.path.append(os.path.abspath(
    os.path.dirname(
        os.path.abspath(__file__)
    )+"/.."
))

extract_metrics = ['train_loss', 'val_loss']


def print_table(data):
    values = [d.values() for d in data]
    if len(data) > 0:
        print(tabulate(values, headers=extract_metrics))
    else:
        print('there is no data')


def main():
    parser = argparse.ArgumentParser(description="lists mlflow's runs")
    run_infos = mlflow.list_run_infos(experiment_id='0')
    data = []
    for run_info in run_infos:
        row = collections.OrderedDict()
        row['id'] = run_info.run_id
        run = mlflow.get_run(run_info.run_id)
        for m in extract_metrics:
            row[m] = run.data.metrics.get(m, None)
        data.append(row)
    print_table(data)


if __name__ == "__main__":
    main()
