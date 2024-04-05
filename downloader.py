import gdown
import os
import zipfile
from argparse import ArgumentParser
import shutil


def download_rn_vit_features():
    # Download the SVM and XGB fea from Google Drive

    rn_vit_features = [
        '1-YIRFjckDsgrlyUDyAx5edRBMZZ4_Ya0',
        '1-YhYbDcdmPMegIApu4DBNnmUyBTvigV5'
    ]

    rn_vit_features_names = [
        'resnet_features.csv',
        'vit_features.csv'
    ]

    dl_folder = 'features'

    if not os.path.exists(dl_folder):
        os.makedirs(dl_folder)

    for file in rn_vit_features:
        url = 'https://drive.google.com/uc?id=' + file
        output = os.path.join(
            dl_folder,
            rn_vit_features_names[rn_vit_features.index(file)])
        gdown.download(url, output, quiet=False)


def download_SVM_XGB_models():
    # Download the SVM and XGB models from Google Drive

    SVM_and_XGB_models = [
        '1-CQPtLhJ87kbOQ6KCVBT5bFdRiEc4NUO',
        '1-1ExkrnS2Vf83coHY7ZOUvKGIhgW3bQj',
        '1-DNF2Twdgn10ogrBSlHVD0xjkg-kTpCU',
        '1-M5dGp2mh77CZOM1HJVNPekZB9GtgNlS',
        '1-1R8ZSbPNtEngMZbLyYPyGJQwqL8NWzT',
    ]

    SVM_and_XGB_models_names = [
        'resnet_svm_model.pkl',
        'resnet_XGB_model.pkl',
        'vit_svm_model.pkl',
        'vit_XGB_model.pkl',
        'resnet_XGB_model_optuna.pkl'
    ]

    dl_folder = 'ML_models'

    if not os.path.exists(dl_folder):
        os.makedirs(dl_folder)

    for file in SVM_and_XGB_models:
        url = 'https://drive.google.com/uc?id=' + file
        output = os.path.join(
            dl_folder,
            SVM_and_XGB_models_names[SVM_and_XGB_models.index(file)])
        gdown.download(url, output, quiet=False)


def download_images():
    # Download images data and unzip

    dl_folder = ''  # root folder

    images_folder = '1-ReG67VFkiPBKNCkynvs7CWwIUqRVPFv'
    folder_name = 'Data_small.zip'
    output = os.path.join(dl_folder, folder_name)
    url = 'https://drive.google.com/uc?id=' + images_folder

    gdown.download(url, output, quiet=False)

    unzip_folder = os.path.join(dl_folder, 'Data_small')

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)

    # Check and rearrange if there's a nested 'Data_small' folder
    nested_folder = os.path.join(unzip_folder, 'Data_small')
    if os.path.exists(nested_folder):
        # Move all files from the nested folder to the intended directory
        for filename in os.listdir(nested_folder):
            shutil.move(os.path.join(nested_folder, filename), unzip_folder)
        # Remove the now empty nested folder
        os.rmdir(nested_folder)


def download_images_HOG_SIFT():
    # Download images data and unzip (Data_small_normalized_adj)

    dl_folder = 'download'

    # Create the folder if it doesn't already exist
    if not os.path.exists(dl_folder):
        os.makedirs(dl_folder)

    images_folder = '1-U7odUujdB5rG8gUguCs0aMt_wMwk4nt'
    folder_name = 'Data_small_normalized_adj.zip'
    output = os.path.join(dl_folder, folder_name)
    url = 'https://drive.google.com/uc?id=' + images_folder

    gdown.download(url, output, quiet=False)

    unzip_folder = os.path.join(dl_folder, 'Data_small_normalized_adj')

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)


def download_RESNET_VIT_weights():

    resnet_and_vit_models = [
        '1-_djdhgvAhgDS_Nxw5qmZaK6q7aqJfr_',
        '1-TnVCNS0VcXuDFUurBct38BvN0FaYpxm'
    ]

    resnet_and_vit_models_names = [
        'resnet50_state_dict.pth',
        'vit_base_patch32_state_dict.pth'
    ]

    dl_folder = 'DL_dicts'

    # Create the folder if it doesn't already exist
    if not os.path.exists(dl_folder):
        os.makedirs(dl_folder)

    for file in resnet_and_vit_models:
        url = 'https://drive.google.com/uc?id=' + file
        output = os.path.join(
            dl_folder,
            resnet_and_vit_models_names[resnet_and_vit_models.index(file)])
        gdown.download(url, output, quiet=False)


def main(selection):
    if selection == 'all':
        download_rn_vit_features()
        download_SVM_XGB_models()
        download_RESNET_VIT_weights()
        download_images()
        download_images_HOG_SIFT()
    elif selection == 'SVM_XGB':
        download_rn_vit_features()
        download_SVM_XGB_models()
    elif selection == 'images':
        download_images()
    elif selection == 'images_HOS_SIFT':
        download_images_HOG_SIFT()
    elif selection == 'RESNET_VIT_weights':
        download_RESNET_VIT_weights()
    else:
        print('Invalid selection')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--selection", type=str, help="object to download", default="all")
    args = parser.parse_args()
    main(args.selection)
