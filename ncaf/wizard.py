#!/usr/bin/env python
# encoding: utf-8

import npyscreen, curses
import logging
import os
import glob

from . import sa_common, queries

selections = {}

log_file = open('/tmp/wizard.log','w+')

def log(*args):
    msg = " ".join(map(str,args)) + '\n'
    log_file.write(msg)
    log_file.flush()

class WizardForm(npyscreen.Form):
    def create(self):
        for how in [
            npyscreen.wgwidget.EXITED_DOWN,
            npyscreen.wgwidget.EXITED_ESCAPE,
            npyscreen.wgwidget.EXITED_RIGHT
        ]:
            self.how_exited_handers[how] = self.exit_application

    def exit_application(self):
        w = self.get_widget("w")
        log("!!!!w value",w.value)
        selections[self.formname] = self.data[w.value]
        log("selections",selections)
        log("self.formname",self.formname)
        FormClass = get_form_class_by_name(self.nextForm)
        self.parentApp.registerForm(self.nextForm, FormClass())
        self.parentApp.setNextForm(self.nextForm)
        self.parentApp.switchFormNow()

class WizardMultiLine(npyscreen.MultiLine):
    def __init__(self,*args,**kwargs):
        if 'select_exit' not in kwargs.keys():
            kwargs['select_exit'] = True
        super().__init__(*args,**kwargs)

class DagRunsForm(WizardForm):

    formname = "MAIN"
    nextForm = 'TASKINSTANCECOUNTS'
    def create(self):
        query = queries.dag_runs()
        log("this is foobar")
        log("dagruns query",str(query))
        keys,self.data = sa_common.fetch_data(query)
        values = [
            " ".join(map(str,curr))
            for curr
            in self.data
        ]
        self.add(
            WizardMultiLine,
            values=values,
            w_id="w"
        )
        super().create()

class TaskInstanceCountsForm(WizardForm):
    formname = "TASKINSTANCECOUNTS"
    nextForm = "TASKINSTANCES"
    def create(self):
        global selections
        run_id = selections['MAIN'][1]
        query = queries.task_state_counts_by_run_id(run_id)
        key, self.data = sa_common.fetch_data(query)
        values = [
            " ".join(map(str,curr[:2]))
            for curr
            in self.data
        ]
        self.add(
            WizardMultiLine,
            values=values,
            w_id="w"
        )
        super().create()


class TaskInstancesForm(WizardForm):
    formname = "TASKINSTANCES"
    nextForm = "TASKINSTANCELOGS"
    def create(self):
        global selections
        run_id = selections['MAIN'][1]
        state = selections['TASKINSTANCECOUNTS'][1]
        query = queries.task_instances_by_state(run_id, state)
        key, self.data = sa_common.fetch_data(query)
        max_taskname_len = max([
            len(curr[0])
            for curr
            in self.data
        ])
        values = [
            f"{curr[0]}{' '*(max_taskname_len-len(curr[0]))}{curr[1]} {curr[2]}"
            for curr
            in self.data
        ]
        self.add(
            WizardMultiLine,
            values=values,
            w_id="w"
        )
        super().create()

class TaskInstanceLogs(WizardForm):
    formname = "TASKINSTANCELOGS"
    nextForm = "TASKINSTANCELOGCONTENT"

    def create(self):
        global selections
        dag_name = selections['MAIN'][0]
        run_id = selections['MAIN'][1]
        run_id_date_part = run_id.split('__')[1]
        task_name = selections['TASKINSTANCES'][0]
        logs_glob = os.path.join(
            os.path.expanduser('~'),
            'airflow',
            'logs',
            dag_name,
            task_name,
            run_id_date_part,
            '*.log'
        )
        filenames = glob.glob(logs_glob)
        filenames = sorted(filenames)
        self.data = filenames
        log("glob",logs_glob)
        log("filenames",filenames)
        log("self.data",self.data)

        values = [
            os.path.basename(curr)
            for curr
            in self.data
        ]
        print('values',values)
        self.add(
            WizardMultiLine,
            values=values,
            w_id="w"
        )
        super().create()


class TaskInstanceLogContent(WizardForm):
    formname = "TASKINSTANCELOGCONTENT"
    nextForm = None

    def create(self):
        global selections
        filename = selections['TASKINSTANCELOGS']
        file = open(filename, mode='r')
        log_contents = file.read()
        self.add(
            npyscreen.MultiLine,
            values=log_contents.split('\n'),
            w_id="w"
        )
        self.data=[]
        super().create()

FormClasses = [
    DagRunsForm,
    TaskInstanceCountsForm,
    TaskInstancesForm,
    TaskInstanceLogs,
    TaskInstanceLogContent
]

def get_form_class_by_name(name):
    for curr in FormClasses:
        if curr.formname == name:
            return curr
    raise Exception(f"could not find form class named {name}")

class MyTestApp(npyscreen.NPSAppManaged):
    def onStart(self):
        MainFormClass = get_form_class_by_name("MAIN")
        self.registerForm("MAIN",MainFormClass())

def main():
    TA = MyTestApp()
    TA.run()


if __name__ == '__main__':
    main()