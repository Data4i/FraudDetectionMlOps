from zenml.pipelines import pipeline
from steps.data_loader import data_loader
from steps.data_cleaner import data_cleaner
from steps.data_splitter import data_splitter
from steps.data_transformer import data_transformer
from steps.model_trainer import model_trainer
from steps.model_evaluator import model_evaluator
from steps.model_tester import model_tester


@pipeline
def full_pipeline(data_path):
    data = data_loader(data_path)
    cleaned_data = data_cleaner(data)
    X_train, X_test, y_train, y_test = data_splitter(cleaned_data)
    transformed_X_train, transformed_X_test = data_transformer(X_train, X_test)
    model = model_trainer(transformed_X_train, y_train)
    f1_evaluation_score = model_evaluator(model, transformed_X_train, y_train)
    f1_test_score = model_tester(model, transformed_X_test, y_test)
