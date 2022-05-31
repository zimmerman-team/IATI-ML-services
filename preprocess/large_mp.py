import tempfile
import json
import os


def write_tmp(data):
    """
    Encodes the data to json and stores it into a temporary file
    :param data:
    :return: the temporary filename
    """
    dirname = 'large_mp/'
    try:
        os.mkdir(os.path.join('/tmp',dirname))
    except FileExistsError:
        pass
    filename = tempfile.mktemp(prefix=dirname)
    with open(filename, 'w+') as f:
        json.dump(data, f)
        f.flush()
    return filename


def read_tmp(filename):
    """
    reads the content of a json-encoded file
    :param filename:
    :return: the decoded data
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def send(ti, data):
    """
    Stores the given data into a temporary file and delivers the
    temporary filename to the receiving task
    :param ti: task id # FIXME: naming??? It's not a task id, but something like a task object!!
    :param data: data to be sent
    :return: None
    """
    tmp_filename = write_tmp(data)
    ti.xcom_push(key='tmp_filename', value=tmp_filename)


def recv(ti, task_id):
    """
    Receives data from another task that was stored into a temporary file
    :param ti: task id
    :param task_id: source task id # FIXME: naming??? and is it really the source task id or the destination task it??
    :return:
    """
    input_filename = ti.xcom_pull(key='tmp_filename', task_ids=task_id)
    assert input_filename is not None
    data = read_tmp(input_filename)
    return data


def clear_recv(ti, task_id):
    """
    Removes the temporary file used for data transfer across tasks
    :param ti: task id
    :param task_id: source task id # FIXME: or destination???
    :return:
    """
    input_filename = ti.xcom_pull(key='tmp_filename', task_ids=task_id)
    os.unlink(input_filename)
