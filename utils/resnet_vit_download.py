import os
import gdown

def get_rn_file():

    dl_folder = 'DL_dicts'
    file_path = 'resnet50_state_dict.pth' 
    full_path = os.path.join(dl_folder, file_path)

    resnet_model = '1-SGB77hy9ybiIP9C9BpH0006u7BhT4r5'
    resnet_name = 'resnet50_state_dict.pth'
    url = 'https://drive.google.com/uc?id=' + resnet_model
    output = os.path.join(dl_folder, resnet_name)
   
    # Check if the file exists
    if not os.path.exists(full_path):
        print(f"File not found. Downloading from Gdrive: {resnet_model}")
        gdown.download(url, output, quiet=False)
    else:
        print(f"No need to download - model already exists at {full_path}")


def get_vit_file():

    dl_folder = 'DL_dicts'
    file_path = 'vit_base_patch32_state_dict.pth'
    full_path = os.path.join(dl_folder, file_path)

    vit_model = '1-TnVCNS0VcXuDFUurBct38BvN0FaYpxm'
    vit_name = 'vit_base_patch32_state_dict.pth'
    url = 'https://drive.google.com/uc?id=' + vit_model
    output = os.path.join(dl_folder, vit_name)
   
    # Check if the file exists
    if not os.path.exists(full_path):
        print(f"File not found. Downloading from Gdrive: {vit_model}")
        gdown.download(url, output, quiet=False)
    else:
        print(f"No need to download - model already exists at {full_path}")