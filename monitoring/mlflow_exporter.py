import flask
import mlflow
from collections import OrderedDict
import re
import glob
import os

port = 5500

app = flask.Flask(__name__)

def extract_run_directory(run):
    chunk = re.sub('^file://','',run.artifact_uri)
    ret = re.sub('/artifacts$','',chunk)
    return ret

def collect_metrics_from_run_directory(directory):
    assert os.path.isdir(directory)
    metrics = OrderedDict()
    metrics_dir = os.path.join(directory,'metrics')
    metrics_filenames = glob.glob(metrics_dir+"/*")
    for metric_filename in metrics_filenames:
        metric_name = os.path.basename(metric_filename)
        with open(metric_filename, 'r') as f:
            last_line = f.readlines()[-1].split(" ")
            metric_ts = int(last_line[0])
            metric_value = float(last_line[1])
            metrics[metric_name] = OrderedDict(ts=metric_ts,value=metric_value)
    return metrics

def collect_metrics():
    collected = []
    for experiment in mlflow.list_experiments():
        for run in mlflow.list_run_infos(experiment.experiment_id):
            directory = extract_run_directory(run)
            run_metrics = collect_metrics_from_run_directory(directory)
            collected.append(OrderedDict(
                experiment_id=experiment.experiment_id,
                experiment_name=experiment.name,
                run_id=run.run_id,
                run_metrics=run_metrics
            ))
    return collected

def run_labels_to_prom(curr_collected, curr_metric_name):
    prom_labels = [
        (other_key, other_val)
        for other_key, other_val
        in curr_collected.items()
        if other_key != 'run_metrics'
    ] + [("metric_name",curr_metric_name)]
    label_strings = [
        label + '="' + str(value) + '"'
        for label, value
        in prom_labels
    ]
    prom_labels_str = ', '.join(label_strings)
    return prom_labels_str

def format_metrics(collected):
    prom_tss = []
    for curr_collected in collected:
        for curr_metric_name, curr_metric_stuff in curr_collected['run_metrics'].items():
            prom_labels_str = run_labels_to_prom(curr_collected,curr_metric_name)
            prom_ts = 'mlflow_metric{' + prom_labels_str + '} '+str(curr_metric_stuff['value'])# + " "+str(curr_metric_stuff['ts'])
            prom_tss.append(prom_ts)
    ret = '\n'.join([
    "# HELP mlflow_metric metrics logged from mlflow's runs",
    "# TYPE mlflow_metric gauge"
    ])+'\n'
    ret += '\n'.join(prom_tss)
    return ret

@app.route('/metrics')
def metrics_endpoint():
    collected_run = collect_metrics()
    formatted_metrics = format_metrics(collected_run)
    return formatted_metrics

def main():
    global port
    app.run(port=port)

if __name__ == "__main__":
    main()