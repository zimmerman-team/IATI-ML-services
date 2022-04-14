FROM alpine:3.14.6
RUN apk add make
RUN apk add go
RUN git clone https://github.com/esnet/gdg
WORKDIR gdg
RUN make
RUN git clone https://github.com/databand-ai/airflow-dashboards
#RUN ./bin/gdg dashboards import airflow-dashboards/grafana/dag-dashboard.json
