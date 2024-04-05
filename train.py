from utils.resnet import prepare_data_resnet, train_resnet, test_resnet
from utils.vit import prepare_data_vit, train_vit, test_vit

from argparse import ArgumentParser


def main(model='vit', image_folder_path='Data_small', eval=False):
    if model == 'resnet':
        print('Selected model: ResNet50. Starting data preparation')
        device, no_of_classes, train_loader, test_loader, dataloader =  \
            prepare_data_resnet(image_folder_path)
        if eval is False:
            print('Data prepared. Starting training')
            no_of_classes, test_loader = \
                train_resnet(device, no_of_classes, train_loader, test_loader)
            print('Training completed. Starting testing')
        else:
            print('Data prepared. Starting testing')
        return test_resnet(no_of_classes, test_loader)

    elif model == 'vit':
        print('Selected model: VIT. Starting data preparation')
        device, no_of_classes, train_loader, test_loader, dataloader = \
            prepare_data_vit(image_folder_path)
        if eval is False:
            print('Data prepared. Starting training')
            no_of_classes, test_loader = \
                train_vit(device, no_of_classes, train_loader, test_loader)
            print('Training completed. Starting testing')
        else:
            print('Data prepared. Starting testing')
        return test_vit(no_of_classes, test_loader)
    else:
        print('Invalid model name')
        return 'Invalid model name'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="select model", default="vit")
    parser.add_argument(
        "--path", type=str, help="image folder path", default='Data_small')
    parser.add_argument(
        "--eval", type=bool, help="evaluate only", default=False)
    args = parser.parse_args()
    main(args.model, args.path, args.eval)
