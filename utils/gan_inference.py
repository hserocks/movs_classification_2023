import gdown
import os
import subprocess
import random

def get_gan_file(model_name = 'cats'):

    dl_folder = 'gan_models'
    
    if model_name == 'cats':
        file_path = 'cats.pkl'
        gan_model = '1TqOqMi34XLw4_0bUmyEo-7tLfDhvGSiV'
        gan_name = 'cats.pkl'
    else:
        file_path = 'cats_dogs.pkl'
        gan_model = '1kZdyJ4LCvY7VTxy-xld6TmP2TnHSL8I8'
        gan_name = 'cats_dogs.pkl'
    
    full_path = os.path.join(dl_folder, file_path)

    
    url = 'https://drive.google.com/uc?id=' + gan_model
    # output = os.path.join(dl_folder, gan_name)

    # Check if the file exists
    if not os.path.exists(dl_folder):
        os.makedirs(dl_folder)
    
    if not os.path.exists(full_path):
        print(f"File not found. Downloading from Gdrive: {gan_model}")
        gdown.download(url, full_path, quiet=False)
    else:
        print(f"No need to download - model already exists at {full_path}")
    
    return full_path



def get_inference(model_name='cats', seed = None):
    gan_path = get_gan_file(model_name)
    full_gan_path = os.path.join('/mnt/volume_lon1_01/Project/movs_classification_2023', gan_path)
    
    # Define the network URL
    network_url = full_gan_path

    # Generate a random seed
    if seed is None:
        random_seed = random.randint(0, 100000)
    else:
        random_seed = seed

    # Path to the shell script
    script_path = "utils/gen_image.sh"

    # Call the shell script with the network URL and random seed as arguments
    subprocess.run([script_path, network_url, str(random_seed)])


def get_last_image(save_folder):
    import glob
    import os

    # Get all jpg images from the folder
    images = glob.glob(os.path.join(save_folder, "*.png"))
    print(images)

    if not images:
        return None  # or you can raise an exception or handle it as needed

    # Sort the images by creation time
    images.sort(key=os.path.getctime)

    # Return the last image
    last_image = images[-1]

    return last_image


if __name__ == '__main__':
    save_folder = 'generated'
    # get_inference()
    get_last_image(save_folder)