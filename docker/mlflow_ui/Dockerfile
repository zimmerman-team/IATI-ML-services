FROM alpine:3.14.6
RUN apk --update add --no-cache g++ lapack-dev
RUN apk add --no-cache python3 py3-pip python3-dev
RUN apk add --no-cache openblas-dev
RUN pip3 install pandas
RUN pip3 install mlflow
ENTRYPOINT mlflow ui