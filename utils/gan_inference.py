import gdown
import os
import subprocess

def get_gan_file():

    dl_folder = 'gan_models'
    file_path = 'cats.pkl'
    full_path = os.path.join(dl_folder, file_path)

    gan_model = '1TqOqMi34XLw4_0bUmyEo-7tLfDhvGSiV'
    gan_name = 'cats.pkl'
    url = 'https://drive.google.com/uc?id=' + gan_model
    output = os.path.join(dl_folder, gan_name)

    # Check if the file exists
    if not os.path.exists(full_path):
        print(f"File not found. Downloading from Gdrive: {gan_model}")
        gdown.download(url, output, quiet=False)
    else:
        print(f"No need to download - model already exists at {full_path}")
    
    return full_path



def get_inference(model_name):
    gan_path = get_gan_file(model_name)
    
    # Define the network URL
    network_url = gan_path

    # Path to the shell script
    script_path = "generate_images.sh"

    # Call the shell script with the network URL as an argument
    subprocess.run([script_path, network_url])

    