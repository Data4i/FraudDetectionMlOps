# import click
# from zenml.client import Client
from pipelines.training import full_pipeline


def main():
    full_pipeline.with_options(config_path="configs/local_config.yaml")()


if __name__ == "__main__":
    main()
    # full_pipeline('data/fraud_detect/PS_20174392719_1491204439457_log.csv')
