MLFlow Exporter
===============

This tool collects the last logged metrics from MLFlow's run directories and publishes them
on a simple web end-point made with Flask.
It needs to be run from the directory containing the `mlflow/` subdirectory.

Then prometheus can be configured to read from this service so that the collected metrics can 
eventually be delivered to tools such as Grafana.

Only the last line of the metrics files is being considered for publishing.

Configuration
-------------

The configuration options are module-wide uppercase variables in the source file `mflow_exporter.py`.
Eventually this might be changed to a more apt configuration system.

Exponentially-Weighted Moving Average (EWMA)
--------------------------------------------

For each metric an exponentially-weighted average is being calculated. 
The default parameter value for the weight of the current metric observation is 0.001.
This methods is used to obtain smoothing of the metric curve.
Also, in the

Web service
-----------

The port is 5500 and the metrics are published on the `/metrics` page.
