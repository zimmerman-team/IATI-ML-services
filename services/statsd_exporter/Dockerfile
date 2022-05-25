FROM alpine:3.14.6
RUN apk add go
RUN apk add git
RUN apk add make
RUN apk add curl
RUN git clone https://github.com/prometheus/statsd_exporter --branch v0.22.5
WORKDIR statsd_exporter
RUN make
RUN go test
ADD statsd_exporter_mapping.conf .
ENTRYPOINT ./statsd_exporter \
  --web.listen-address="0.0.0.0:9102" \
  --statsd.listen-udp="0.0.0.0:9125" \
  --statsd.listen-tcp="0.0.0.0:9125" \
  --statsd.mapping-config=statsd_exporter_mapping.conf
