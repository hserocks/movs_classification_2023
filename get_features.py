from utils.resnet import prepare_data_resnet, extract_features_resnet
from utils.vit import prepare_data_vit, extract_features_vit

from argparse import ArgumentParser

def main(model, image_folder_path):
    if model == 'resnet':
        print('Selected model: ResNet50. Starting data preparation')
        device, no_of_classes, train_loader, test_loader, dataloader = prepare_data_resnet(image_folder_path)
        print('Data prepared. Starting to extract features')
        features, labels, file_path = extract_features_resnet(device, no_of_classes, dataloader)
        print(f'Features extracted and saved here: {file_path}')

    elif model == 'vit':
        print('Selected model: VIT. Strating data preparation')
        device, no_of_classes, train_loader, test_loader, dataloader = prepare_data_vit(image_folder_path)
        print('Data prepared. Starting to extract features')
        features, labels, file_path = extract_features_vit(device, no_of_classes, dataloader)
        print('Training completed. Starting testing')
        print(f'Features extracted and saved here: {file_path}')
    else:
        print('Invalid model name')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="select model", default="vit")
    parser.add_argument(
        "--path", type=str, help="image folder path", default='Data_small')
    args = parser.parse_args()
    main(args.model, args.path)