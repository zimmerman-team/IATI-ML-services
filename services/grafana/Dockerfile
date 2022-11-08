FROM grafana/grafana-oss
ADD dashboards_provisioning.yaml /etc/grafana/provisioning/dashboards
ADD grafana_mlflow_dashboard.json /etc/grafana/provisioning/dashboards
USER root
RUN apk add git
RUN git clone https://github.com/databand-ai/airflow-dashboards
RUN cp -vf airflow-dashboards/grafana/*.json /etc/grafana/provisioning/dashboards
#RUN chown grafana . -Rvf
USER grafana
