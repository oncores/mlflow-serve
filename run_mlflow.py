import mlflow

mlflow.set_tracking_uri('http://35.78.221.92:5000/')
mlflow.start_run(experiment_id=2, run_name='elasticnet_dev')
print(mlflow.get_tracking_uri())
print(mlflow.get_artifact_uri())