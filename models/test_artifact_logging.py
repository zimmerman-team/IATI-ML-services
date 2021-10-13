import mlflow

mlflow.set_experiment("test_artifact_logging")

mlflow.start_run(run_name="test_artifact_logging")
mlflow.log_artifact(__file__)