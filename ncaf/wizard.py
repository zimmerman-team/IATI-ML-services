#!/usr/bin/env python
# encoding: utf-8

import npyscreen, curses
import logging
import os
import glob
from collections import OrderedDict

from . import sa_common, queries

selections = OrderedDict()

log_file = open('/tmp/wizard.log','w+')

def log(*args):
    msg = " ".join(map(str,args)) + '\n'
    log_file.write(msg)
    log_file.flush()

class WizardForm(npyscreen.Form):
    def create(self):
        for how in [
            npyscreen.wgwidget.EXITED_DOWN,
            npyscreen.wgwidget.EXITED_RIGHT
        ]:
            self.how_exited_handers[how] = self.exit_application

        for how in [
            npyscreen.wgwidget.EXITED_ESCAPE
        ]:
            self.how_exited_handers[how] = self.get_back

    def exit_application(self):
        log(self.formname,'exit_application')
        w = self.get_widget("w")
        selections[self.formname] = self.data[w.value]
        FormClass = get_form_class_by_name(self.nextForm)
        self.parentApp.registerForm(self.nextForm, FormClass())
        self.parentApp.setNextForm(self.nextForm)
        self.parentApp.switchFormNow()

    def get_back(self):
        prev = forms_flow[self.formname]['prev']
        log(self.formname,"get_back to",prev)
        FormClass = get_form_class_by_name(prev)
        #self.parentApp.registerForm(self.nextForm, FormClass())
        self.parentApp.setNextForm(prev)
        self.parentApp.switchFormNow()

    def set_header(self):
        header_strs = []
        for formname, selection in selections.items():
            if formname == self.formname:
                break

            descriptive_field_idx = getattr(get_form_class_by_name(formname),'descriptive_field_idx',0)

            header_strs.append(str(selection[descriptive_field_idx]))

        header_str = f"{self.formname} | " + " / ".join(header_strs)
        self.add(
            npyscreen.FixedText,
            value=header_str,
            w_id="header",
            editable=False,
            color='IMPORTANT'
        )


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
        keys,self.data = sa_common.fetch_data(query)
        values = [
            " ".join(map(str,curr))
            for curr
            in self.data
        ]
        self.set_header()
        self.add(
            WizardMultiLine,
            values=values,
            w_id="w"
        )
        super().create()

class TaskInstanceCountsForm(WizardForm):
    formname = "TASKINSTANCECOUNTS"
    nextForm = "TASKINSTANCES"
    descriptive_field_idx = 1
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
        self.set_header()
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
            f"{curr[0]}{' '*(1+max_taskname_len-len(curr[0]))}{curr[1]} {curr[2]}"
            for curr
            in self.data
        ]
        self.set_header()
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
        self.set_header()
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
        self.set_header()
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

forms_flow = OrderedDict()
prev = None
for FormClass in FormClasses:
    next = FormClass.nextForm
    curr = FormClass.formname
    forms_flow[curr] = dict(
        prev=prev,
        curr=curr,
        next=next
    )
    prev = curr

log("forms_flow",forms_flow)

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