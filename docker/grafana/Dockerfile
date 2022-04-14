FROM grafana/grafana-oss

USER root
RUN apk add git
RUN apk add make
RUN apk add go
RUN git clone https://github.com/esnet/gdg
WORKDIR gdg
RUN make
RUN git clone https://github.com/databand-ai/airflow-dashboards
RUN chown grafana . -Rvf
#RUN ./bin/gdg dashboards import airflow-dashboards/grafana/dag-dashboard.json
USER grafana
#ENTRYPOINT sh