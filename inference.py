from utils.short_model import get_categories_vit, get_categories_rn
from utils.inference_svm_xgb import get_categories_SVM, get_categories_XGB
from argparse import ArgumentParser

# import os
# from PIL import Image
import requests
from io import BytesIO


def open_image(path_or_url):
    # Check if the string is a URL
    if path_or_url.startswith(('http://', 'https://')):
        response = requests.get(path_or_url)
        response.raise_for_status()
        # Open the image directly from the response's bytes
        return BytesIO(response.content)
    else:
        return path_or_url


def main(model, path):
    if model == 'resnet':
        print('Selected model: ResNet50. Starting inference')
        image = open_image(path)
        return get_categories_rn(image)

    elif model == 'vit':
        print('Selected model: VIT. Starting inference')
        image = open_image(path)
        return get_categories_vit(image)

    elif model == 'resnet_svm':
        print('Selected model: ResNet50 with SVM. Starting inference')
        image = open_image(path)
        return get_categories_SVM('resnet', image)

    elif model == 'vit_svm':
        print('Selected model: VIT with SVM. Starting inference')
        image = open_image(path)
        return get_categories_SVM('vit', image)

    elif model == 'resnet_xgb':
        print('Selected model: ResNet50 with XGB. Starting inference')
        image = open_image(path)
        return get_categories_XGB('resnet', image)

    elif model == 'vit_xgb':
        print('Selected model: VIT with XGB. Starting inference')
        image = open_image(path)
        return get_categories_XGB('vit', image)

    else:
        print('Invalid model name')
        return 'Invalid model name'

    # return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="select model", default="vit")
    parser.add_argument("--path", type=str, help="image path",
                        default = 'https://cdn.britannica.com/55/174255-050-526314B6/brown-Guernsey-cow.jpg')
    args = parser.parse_args()
    main(args.model, args.path)
