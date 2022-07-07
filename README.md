# Learning sets

Models to learn latent code representation of sets being the one2many relationships within the activities (for example 
the activity->budget relationship). This will allow those fields to be replaced to fixed-length meaningful representation
codes which ultimately will help to have fixed-length vectors of activites, for supervised (for example classification) 
or unsupervised learning (learning latent code representations of activities themselves)

## Configuration

The example configuration file `configs/example.yaml` contains comments on every configuration option.
Adjust them accordingly and rename it to `<yourhostname>.yaml`. This way the configurator component is going to
read a different configuration file according to which machine it's been deployed.

## Deployment

The system is configured for an easy deployment on unix systems via docker.

Launching `bash rsync.sh` will copy all relevant files and directories to the remote machine.

Please read `services/README.md` for details on how to get all services up and running.

Within the Airflow interface, then accessible via http://127.0.3.1:8080 , one can launch the data download preprocessing DAG
(`Downloads sets data from IATI.cloud`) and the various model training DAGs
(`train_models_dag_*`).

## Runs

Everything is run via Airflow. Its interface is accessible via http://127.0.3.1:8080

 * to trigger the preprocessing and data preparation launch the `download_and_preprocess_sets` DAG
 * to train the deep set model on relational fields launch the `train_models_dag_(i)dspn_autoencoder` DAG
 * to create the fixed-length-datapoints activity dataset launch the `vectorize_activities` DAG
 * to train the main activity autoencoder launch the `train_models_dag_activity_autoencoder` DAG

## Independent subcomponents

The codebase has been split in independent repositories that handle different aspects of the workflow. They can be 
categorized into two groups: Executables and Libraries.

### Executables

These are tools that can be independently run and they fulfil a specific task.

#### Mlflow_exporter

Delivers mlflow statistics, which are supposed to be generated in the `mlflow/` directory. 
`mlflow_exporter` delivers the statistics to Prometheus, which will deliver them  to Grafana for model training 
status visualization.

Please read `services/mflow_exporter/README.md` for details.

#### Ncaf

ncurses-based airflow task exploration tool is a text-based wizard/menu interface to easily have an overview of which
Airflow DAGs are being run and the status of the tasks.

Please read `ncaf/README.md` for details.

### Libraries

#### Configurator

It's a module that allows for easy configuration across multiple systems.
The configuration files are stored in the `configs/` directory.

Please read `configurator/README.md` for details.

#### Dataspecs

It's a library that allows for object-oriented data mapping representation. Useful for data download, preprocessing
and preparation of the numpy arrays that are used for machine learning.

Please read `dataspecs/README.md` for details.

#### Niftycollection

Niftycollection is a user-friendly way to access a dictionary, also allows for automatic indexing
for objects that have a `name` attribute.

Please read `niftycollection/README.md` for details.

#### Chunking_dataset

Chunking_dataset is a library based on pytorch-lightning that allows for splitting a training epoch in smaller chunks.

Please read `chunking_dataset/README.md` for details.

#### Large_mp

Large_mp is a library built to circumvent the fact that the size of messages passed between Airflow tasks is limited.

Please read `large_mp/README.md` for details.