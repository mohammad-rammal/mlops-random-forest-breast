import pandas as pd
import numpy as np
import joblib
import mlflow

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

mlflow.set_tracking_uri(uri="http://mlflow:5000")
mlflow.set_experiment("ML OPS")
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name="randomForest_cv10Fold") as run :
    # importing the dataset
    data=pd.read_csv("../data-storage/breast_dataset.csv")

    # preprocessing (adjusting) the data
    label_class = LabelEncoder()
    data["Class"] = label_class.fit_transform(data["Class"])


    X = data.iloc[:, 0:9]
    y = data.iloc[:, 9]
    X = X.astype({col: 'float64' for col in X.select_dtypes(include='int').columns})

    # create validation set to validate the model at the end (unseen)
    X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size=0.2,random_state=17)

    # create model for ML with random forest
    random_forest = RandomForestClassifier()

    # run K-fold validation where k=10 or 3
    # cv=cross validate
    # return_estimator = return model which made in it cv (return classifier), return models which make at them train at 10 (not just values but actual model)
    # for model training
    # cv=10 , divide X_train, y_train to 10 parts (potion) loop 1test and 9train 10 times with different parts from X_train and y_train
    # based on scoring of accuracy
    cv_results=cross_validate(
      random_forest,
      X_train,
      y_train,
      cv=10,
      return_estimator= True,
      scoring="accuracy"
    )

    # scores = accuracy at testing
    scores = cv_results["test_score"]
    print(scores)
    #[0.98181818 0.96363636 0.98181818 0.98181818 0.98181818 0.98181818 0.96296296 0.98148148 0.98148148 0.96296296]

    mean = scores.mean()
    std = scores.std()

    mlflow.log_metric("cv_accuracy_mean",mean)
    mlflow.log_metric("cv_accuracy_std",std)

    print("Mean: ", mean, " , Standard Deviation: ",std)
    # Mean:  0.97616   , Standard Deviation:  0.00849 

    # my model is stable over any dividing the dataset
    ###################################################

    # find index which got max score
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_model = cv_results["estimator"][best_idx]

    print("Best Model Score: ", best_score)

    y_validation_pred = best_model.predict(X_validation)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_validation,y_validation_pred))
    print("Classification Report: ")
    print(classification_report(y_validation,y_validation_pred))

    mlflow.sklearn.log_model (
        sk_model = best_model,
        artifact_path = "model",
        registered_model_name = "BreastCancer_RF",
        serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        signature = mlflow.models.signature.infer_signature(X_validation,y_validation_pred),
        input_example = X_validation.head(1)
    )

    # save the model as joblib
    # joblib.dump(best_model, "../model-storage/forest.joblib")