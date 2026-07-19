from dataclasses import dataclass
from time import perf_counter
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    macro_f1: float
    training_time_seconds: float

def build_candidate_models(random_state=42):
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state))
        ]),
        "random_forest": RandomForestClassifier(n_estimators=220, class_weight="balanced", random_state=random_state, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

def evaluate_models(x_train, x_test, y_train, y_test, random_state=42):
    results=[]
    for name, base in build_candidate_models(random_state).items():
        model=clone(base); t0=perf_counter(); model.fit(x_train,y_train); dt=perf_counter()-t0
        pred=model.predict(x_test)
        results.append(ModelResult(name, model, float(accuracy_score(y_test,pred)), float(f1_score(y_test,pred,average="macro",zero_division=0)), float(dt)))
    best=max(results,key=lambda r:(r.macro_f1,r.accuracy))
    return results,best

def results_to_dataframe(results):
    return pd.DataFrame([{"model":r.name,"accuracy":r.accuracy,"macro_f1":r.macro_f1,"training_time_seconds":r.training_time_seconds} for r in results]).sort_values(["macro_f1","accuracy"],ascending=False).reset_index(drop=True)
