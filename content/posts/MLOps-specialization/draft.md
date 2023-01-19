
## Course 2: Machine Learning Data Lifecycle in Production

### Week 1: Collecting, Labeling, and Validating data

#### Overview

Production ML systems have specific challenges compared to an academic or research setting:
* Changing data: Continuous  
* Cost / Performance / scalability
* Interpretability

academic vs prod:

data: static vs dynamic
priority: accuracy vs fast inference / good interpretability / accuracy / cost
model training: continuously assess and retrain
challenge: high accuray vs entire system

labeling
feature space coverage: sample represents all cases that will happens in production

minimal dimensionality: for perf
max predictive data
fairness
rare conditions

software dev: scalabality / extensibility / configuration / consistency & reproductibility / safety & securty / modularity / testability / monitoring

challenges in prod ML:
handle continuously chanching data
optimize compute resource costs

---------

#### ML Pipelines

DAGS Directed Acyclic graph 

Pipeline Orchestraton Frameworks: Argo : Airflow / Celery

TensorFlox Extended TFX

#### Collecting data

maximize predictive content
remove non-informative data
feature space coverage

### labeling Data

résumé: il faut monitorer et retrainer

process feedback (direct labeling): actual vs predicted click-through


human labeling
semi supervised
active learning

direct labeling: features from inference requests


===

