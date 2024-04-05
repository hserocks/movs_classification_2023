import os
from utils.svm_xgb import (
    train_svm, eval_svm, load_features, train_xgb, eval_xgb)
from argparse import ArgumentParser
# from utils.resnet import prepare_data_resnet, train_resnet, test_resnet
# from utils.vit import prepare_data_vit, train_vit, test_vit


def main(features, model, eval):
    if features == 'resnet':
        f_name = 'ResNet50'
    elif features == 'vit':
        f_name = 'VIT'
    else:
        print('Invalid features name')
        return
    if model == 'svm':
        m_name = 'SVM'
    elif model == 'xgb':
        m_name = 'XGB'
    else:
        print('Invalid model name')
        return

    print(f'Selected features: {f_name}. Selected model: {m_name}. Starting->')
    X_train, X_test, y_train, y_test = load_features(features)
    if eval is False:
        if model == 'svm':
            svm_pipeline, model_path, X_test, y_test =  \
                train_svm(X_train, X_test, y_train, y_test, model)
            print(f'Training completed, model saved here: {model_path}')
            print('Starting testing')
            metrics_dict = eval_svm(model_path, X_test, y_test)
            print('Complete')
            return metrics_dict

        elif model == 'xgb':
            model_path, X_test, y_test = \
                train_xgb(X_train, X_test, y_train, y_test, model)
            print(f'Training completed, model saved here: {model_path}')
            print('Starting testing')
            metrics_dict = eval_xgb(model_path, X_test, y_test)
            print('Complete')
            return metrics_dict
    else:
        if model == 'svm':
            if features == 'resnet':
                file_name = 'resnet_svm_model.pkl'
            elif features == 'vit':
                file_name = 'vit_svm_model.pkl'

            model_folder = 'ML_models'
            model_path = os.path.join(model_folder, file_name)
            print('Starting testing')
            metrics_dict = eval_svm(model_path, X_test, y_test)
            print('Complete')
            return metrics_dict

        elif model == 'xgb':
            if features == 'resnet':
                file_name = 'resnet_xgb_model.pkl'
            elif features == 'vit':
                file_name = 'vit_xgb_model.pkl'

            model_folder = 'ML_models'
            model_path = os.path.join(model_folder, file_name)
            print('Starting testing')
            metrics_dict = eval_xgb(model_path, X_test, y_test)
            print('Complete')
            return metrics_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--features", type=str, help="select features", default="vit")
    parser.add_argument(
        "--model", type=str, help="select model", default='svm')
    parser.add_argument(
        "--eval", type=bool, help="evaluate only", default=False)
    args = parser.parse_args()
    main(args.features, args.model, args.eval)
