Commands
========

To be run in this directory: 
    
    make build
    make up

What it does is it builds and runs all docker instances and is put in the background.

Other options are:
    
    make status
    
Which shows the running status of the services.
    
    make airflow_up

Just launches airflow-related services

To stop everything run:

    make down

To see the status of the various running docker containers run:
    
    make status

This will also show the port mapping from inside the container to the outside world and
will help you diagnose issues.


Advantages of dockerization
===========================

- no need to specify specific listening hostnames in order to restrict access
- doesn't mess with hosting machine's installation and libraries
- replicable install of the services
- easy tear down of a service
- existing ecosystem to monitor running docker instances
- easy migration on a different machine
- straightforward multiple instances of a service. Useful for load balancing (kubernetes..)

Services
========

mongo
-----

Stores IATI activity data in various form according to subsequent preprocessing stages.

statsd_exporter
---------------

prometheus
----------

Collects various statistics from various services' states that will be delivered to grafana.

grafana
-------

Shows charts of the various statistics that have been collected from prometheus.

mlflow_ui
---------

Shows all the ML models' training plots, with all statistics, such
as training and validation error.

mlflow_exporter
---------------

Collects metrics from mlflow and exposes them to an http page for prometheus.

airflow_webserver
-----------------

It's the web interface to airflow.

airflow_scheduler
-----------------

Schedules the DAGs run with tasks for preprocessing and model training. 

postgresql_for_airflow
----------------------

Persistency data for airflow.
