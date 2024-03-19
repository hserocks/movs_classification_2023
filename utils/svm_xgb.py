from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import os
import pandas as pd

import xgboost as xgb

def load_features(features):
    
    if features == 'resnet':
        file_name = 'resnet_features.csv'
    elif features == 'vit':
        file_name = 'vit_features.csv'
    else:
        print('Invalid features name')
        return
    
    feature_folder = 'features'
    file_path = os.path.join(feature_folder, file_name)
   
    print('Loading features')
    df = pd.read_csv(file_path)
    #df = pd.read_csv('extracted_features_resnet2.csv')
    X = df.drop('label', axis=1).values  # Features
    y = df['label'].values  # Labels

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Features loaded')

    return X_train, X_test, y_train, y_test


def train_svm(X_train, X_test, y_train, y_test, model_name):
    # Create a pipeline that includes scaling and the SVM
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear'))

    # Train the SVM
    svm_pipeline.fit(X_train, y_train)

    # Save the model to disk
    if model_name == 'resnet':
        file_name = 'resnet_svm_model.pkl'
    elif model_name == 'vit':
        file_name = 'vit_svm_model.pkl'
    else:
        print('Invalid model name')
        return
    
    model_folder = 'ML_models'
    model_path = os.path.join(model_folder, file_name)

    with open(model_path, 'wb') as file:
        pickle.dump(svm_pipeline, file)

    return svm_pipeline, model_path, X_test, y_test
    
def eval_svm(model_path, X_test, y_test):
    # Evaluate the model
    with open(model_path, 'rb') as file:
        svm_pipeline = pickle.load(file)

    predictions = svm_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}")
    print(f"Recall: {recall*100:.2f}")
    print(f"F1 Score: {f1*100:.2f}")
    
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy
    metrics_dict['precision'] = precision
    metrics_dict['recall'] = recall
    metrics_dict['f1'] = f1
    
    return metrics_dict



def train_xgb(X_train, X_test, y_train, y_test, model_name):

    params = {
        #'objective': 'multi:softmax',  # Use 'multi:softprob' for probability predictions
        'objective': 'multi:softprob',
        #'num_class': len(set(final_labels)),  # Set to the number of classes
        'num_class': 81,  # Set to the number of classes
        'device': 'cuda',  # Use GPU acceleration # to replace with 'device' = 'CUDA' for GPU
        'eval_metric': 'mlogloss',  # Multiclass log loss
        'learning_rate': 0.05,
        # Add more parameters here based on your requirements
    }
    

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create DMatrix for training and validation sets
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    num_round = 100  # Number of boosting rounds

    eval_set = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(params, dtrain, num_boost_round=num_round, evals=eval_set, early_stopping_rounds=20, verbose_eval=True)


    # Save the model to disk
    if model_name == 'resnet':
        file_name = 'resnet_XGB_model.pkl'
    elif model_name == 'vit':
        file_name = 'vit_XGB_model.pkl'
    else:
        print('Invalid model name')
        return
    
    model_folder = 'ML_models'
    model_path = os.path.join(model_folder, file_name)
  
    with open(model_path, 'wb') as file:
        pickle.dump(bst, file)
    
    return model_path, X_test, y_test

def eval_xgb(model_path, X_test, y_test):

    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    dtest = xgb.DMatrix(X_test, label=y_test)
    predictions = loaded_model.predict(dtest).argmax(axis=1)
    

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test f1 score: {f1*100:.2f}%")
    
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy
    metrics_dict['f1'] = f1
    
    return metrics_dict