Ncurses-based Airflow Tasks Exploration Tool
============================================

This tool uses a wizard textual interface to get summaries of tasks in Airflow and to read their logs.

It is specifically built for Airflow configurations using PostgreSQL and log files.

Launch
------

If it's part of a bigger project and deployed in the `ncaf/` directory, then launch with this
command:

`python3 -m ncaf.wizard`

Otherwise, just `python3 wizard.py` in its directory.

Configuration
-------------

This step is necessary for this tool's functioning.
A `configurator` instance needs to be deployed either in the parent directory of `ncaf/` or in `ncaf/`
itself.
`configurator`'s config needs to have a `get_airflow_sqlalchemy_conn()` method returning a SQLAlchemy
connection string to Airflow's postgresql.

Wizard's structure
------------------

The wizard is composed of 5 subsequent menu pages:

 * MAIN: shows the list of DAGs
 * TASKINSTANCECOUNTS: for every DAG status type it shows the count of how many runs are in that status
   (which is, for example, `running` or `success`)
 * TASKINSTANCES: lists the individual task instances that have been run
 * TASKINSTANCELOGS: lists the log files produced by the selected task instance
 * TASKINSTANCELOGCONTENT: shows the content of the log file that has been selected