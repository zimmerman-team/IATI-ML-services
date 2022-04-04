#!/usr/bin/env python
# encoding: utf-8

import npyscreen, curses
import logging

from . import dag_runs, sa_common, task_state_counts

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
        selections[self.name] = self.data[w.value]
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
    name = "MAIN"
    nextForm = 'TASKINSTANCE'
    def create(self):
        query = dag_runs.query
        log("this is foobar")
        log("dagruns query",str(query))
        keys,self.data = sa_common.fetch_data(query)
        values = [
            " ".join(curr)
            for curr
            in self.data
        ]
        self.add(
            WizardMultiLine,
            values=values,
            w_id="w"
        )
        super().create()

class TaskInstanceForm(WizardForm):
    name = "TASKINSTANCE"
    nextForm = None
    def create(self):
        global selections
        log("hello there keys",selections.keys())
        run_id = selections['MAIN'].split(" ")[1]
        log("HALLO run_id",run_id)
        query = task_state_counts.query_by_run_id(run_id)
        log("HOLA query",query)
        key, self.data = sa_common.fetch_data(query)
        values = [
            " ".join(curr)
            for curr
            in self.data
        ]
        self.add(
            WizardMultiLine,
            values=values,
            w_id="w"
        )
        super().create()


FormClasses = [
    DagRunsForm,
    TaskInstanceForm
]

def get_form_class_by_name(name):
    for curr in FormClasses:
        if curr.name == name:
            return curr
    raise Exception(f"could not find form class named {name}")

class MyTestApp(npyscreen.NPSAppManaged):
    def onStart(self):
        MainFormClass = get_form_class_by_name("MAIN")
        self.registerForm("MAIN",MainFormClass())
        pass#for FormClass in FormClasses:
            #self.registerForm(FormClass.name, FormClass())


def main():
    TA = MyTestApp()
    TA.run()


if __name__ == '__main__':
    main()