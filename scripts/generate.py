import sys
import os
from random import randint
from argparse import ArgumentParser

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gan_inference import get_inference, get_last_image


def main(model, seed):
    if model == 'cats':
        if seed is None:
            random_seed = randint(0, 100000)
            print(f"Нет аргументов, генерируем случайного кота, seed = {random_seed} (<1 мин.)")
            get_inference(seed = random_seed, model_name='cats')
        else:
            try:
                seed = int(seed)
                print(f"Генерируем кота с seed = {seed} (<1 мин.)")
                get_inference(seed = seed, model_name=model)
            except ValueError:
                random_seed = randint(0, 100000)
                print(f"Неверный аргумент, генерируем случайного кота, seed = {random_seed}  (<1 мин.)")
                get_inference(seed = random_seed, model_name='cats')
    else:
        if seed is None:
            random_seed = randint(0, 100000)
            print(f"Нет аргументов, генерируем случайного кота, seed = {random_seed} (<1 мин.)")
            get_inference(seed = random_seed, model_name='cats')
        else:
            try:
                seed = int(seed)
                print(f"Генерируем кота с seed = {seed} (<1 мин.)")
                get_inference(seed = seed, model_name=model)
            except ValueError:
                random_seed = randint(0, 100000)
                print(f"Неверный аргумент, генерируем случайного кота, seed = {random_seed}  (<1 мин.)")
                get_inference(seed = random_seed, model_name='cats')
        
    generated_image = get_last_image('generated')
    print(generated_image)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="select model", default="cats")
    parser.add_argument(
        "--seed", type=str, help="seed", default=None)
    args = parser.parse_args()
    main(model = args.model, seed = args.seed)
