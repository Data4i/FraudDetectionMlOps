# import click
# from zenml.client import Client
from pipelines.training import full_pipeline


    

if __name__ == "__main__":
    full_pipeline('data/fraud_detect/PS_20174392719_1491204439457_log.csv')