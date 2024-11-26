import os 
import joblib 
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.artifact_stores.base_artifact_store import BaseArtifactStore
from typing import Optional, Any, Type
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin

MODEL_FILENAME = 'decision_tree_model.pkl'

class DecisionTreeClassifierMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = [DecisionTreeClassifier, ClassifierMixin]
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL
    
    def __init__(self, uri: str, artifact_store: Optional[BaseArtifactStore] = None):
        # super().__init__(uri, artifact_store)
        self.uri = uri
        self._artifact_store = artifact_store
        
    def load(self, data_type):
        filepath = os.path.join(self.uri, MODEL_FILENAME)
        return joblib.load(filepath)
        
    
    def save(self, data_type: Any) -> Any:
        filepath = os.path.join(self.uri, MODEL_FILENAME)
        joblib.dump(data_type, filepath)
