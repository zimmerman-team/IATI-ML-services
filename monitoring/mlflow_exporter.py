import flask
import mlflow
from collections import OrderedDict
import re
import glob
import os
import sys
import time
import logging
port = 5500

app = flask.Flask(__name__)

# if a metric has been updated in the last 30 minutes, then a run is detected as "running"
RUNNING_DETECTION_TIMEDELTA = 60 * 30

def log(*args):
    msg = " ".join(map(str,args))
    logging.debug(msg)

def extract_run_directory(run):
    chunk = re.sub('^file://','',run.artifact_uri)
    ret = re.sub('/artifacts$','',chunk)
    return ret

def collect_params_from_run_directory(directory):
    """
    A few selected parameters end up as prometheus metric labels
    :param directory: directory of the mlflow run
    :return: OrderedDict of selected parameters and values
    """
    assert os.path.isdir(directory)
    params = OrderedDict(
        rel_name=None,
        config_name=None
    ) # selecting just a subset of parameters to be added as prometheus labels
    params_dir = os.path.join(directory,'params')
    for param_name in params.keys():
        param_filename = os.path.join(params_dir,param_name)
        value = None # default if param is missing
        if os.path.isfile(param_filename):
            with open(param_filename, 'r') as f:
                value = f.readline() # usually there is only one line in a param file
                value = value.lstrip().rstrip() # remove eventual spaces and newlines
                params[param_name] = value
    return params

def collect_metrics_from_run_directory(directory):
    global RUNNING_DETECTION_TIMEDELTA
    assert os.path.isdir(directory)
    running = False # becomes true if a metric has been updated recently
    metrics = OrderedDict()
    metrics_dir = os.path.join(directory,'metrics')
    metrics_filenames = glob.glob(metrics_dir+"/*")
    for metric_filename in metrics_filenames:
        metric_name = os.path.basename(metric_filename)
        with open(metric_filename, 'r') as f:
            last_line = f.readlines()[-1].split(" ")
            metric_ts = int(last_line[0][:10]) # just keeping the seconds part of the timestamp
            diff = abs( metric_ts - time.time() )
            log("metric_ts",metric_ts,"time.time()",time.time(),"diff",diff)
            if diff < RUNNING_DETECTION_TIMEDELTA:
                running = True
            metric_value = float(last_line[1])
            metrics[metric_name] = OrderedDict(ts=metric_ts,value=metric_value)
    return metrics, running

def collect_metrics():
    collected = []
    for experiment in mlflow.list_experiments():
        for run in mlflow.list_run_infos(experiment.experiment_id):
            directory = extract_run_directory(run)
            run_metrics, running = collect_metrics_from_run_directory(directory)
            run_info_metrics = OrderedDict(
                start_time=run.start_time,
                end_time=run.end_time
            )
            run_metrics.update(run_info_metrics)
            curr = OrderedDict(
                experiment_id=experiment.experiment_id,
                experiment_name=experiment.name,
                run_id=run.run_id,
                run_metrics=run_metrics,
                running=int(running) # prometheus does not have booleans
            )
            log("run_id",run.run_id,"running",running)
            selected_params = collect_params_from_run_directory(directory)
            curr.update(selected_params)
            collected.append(curr)
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
            if type(curr_metric_stuff) != OrderedDict:
                if type(curr_metric_stuff) not in (float,int):
                    curr_metric_stuff = float('nan')
                curr_metric_stuff = OrderedDict(
                    value=curr_metric_stuff,
                    ts=float("nan")
                )
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
    logging.basicConfig(level=logging.INFO)
    if not os.path.isdir('mlruns'):
        logging.error("mlruns/ dir not detected in current working directory")
        sys.exit(-1)
    app.run(port=port,debug=True)

if __name__ == "__main__":
    main()