# import click
from zenml.client import Client
from pipelines.training import full_pipeline

client = Client()
active_stack = client.active_stack
mlflow_tracker = active_stack.experiment_tracker
if mlflow_tracker.flavor == 'mlflow':
    tracking_url = mlflow_tracker.get_tracking_uri()
else:
    print('tracker not available')
# tracking_url = trainer_step.run_metadata["experiment_tracker_url"].value
print(tracking_url)

def main():
    full_pipeline.with_options(config_path="configs/local_config.yaml")()


if __name__ == "__main__":
    main()
    # full_pipeline('data/fraud_detect/PS_20174392719_1491204439457_log.csv')
