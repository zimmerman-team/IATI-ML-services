import tempfile
import json
import os

def write_tmp(data):
    filename = tempfile.mktemp()
    with open(filename, 'w+') as f:
        json.dump(data, f)
        f.flush()
    return filename

def read_tmp(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def send(ti, data):
    tmp_filename = write_tmp(data)
    ti.xcom_push(key='tmp_filename', value=tmp_filename)

def recv(ti,task_id):
    input_filename = ti.xcom_pull(key='tmp_filename', task_ids=task_id)
    assert input_filename is not None
    data = read_tmp(input_filename)
    return data

def clear_recv(ti, task_id):
    input_filename = ti.xcom_pull(key='tmp_filename', task_ids=task_id)
    os.unlink(input_filename)

