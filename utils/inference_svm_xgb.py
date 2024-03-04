import torchvision.models as models
import torch.nn as nn

from torchvision import transforms
from PIL import Image
import json


import numpy as np
from tqdm import tqdm

import pickle
import timm
import torch

import pickle
import xgboost as xgb
import pandas as pd
import os


def get_features(feature_model, new_image_path):
    if feature_model == 'resnet':
        features = image_to_features_resnet(new_image_path)
    
    elif feature_model == 'vit':
        features = image_to_features_vit(new_image_path)

    else:
        print('Invalid features name')
        return
    return features

def get_categories_SVM(feature_model, new_image_path):
    # First, recreate the model architecture
    if feature_model == 'resnet':
        model_path = 'resnet_svm_model.pkl'
    
    elif feature_model == 'vit':
        model_path = 'vit_svm_model.pkl'
    else:
        print('Invalid features name')
        return

    features = get_features(feature_model, new_image_path)

    folder = 'ML_models'
    full_path = os.path.join(folder, model_path)

    # Load the trained SVM model from the pickle file
    with open(full_path, 'rb') as file:
        svm_model = pickle.load(file)

    # Load category names
    with open('class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)

    # Get class names for prediction index
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Make predictions
    prediction = svm_model.predict(features)
    predicted_class_name = idx_to_class[prediction[0]]
    print(f'The model thinks it is: {predicted_class_name}')


def get_categories_XGB(feature_model, new_image_path):
    
    if feature_model == 'resnet':
        model_path = 'resnet_XGB_model.pkl'
    elif feature_model == 'vit':
        model_path = 'vit_XGB_model.pkl'
    else:
        print('Invalid features name')
        return
    
    features = get_features(feature_model, new_image_path)

    folder = 'ML_models'
    full_path = os.path.join(folder, model_path)
    
    with open(full_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # Load category names
    with open('class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)

    # Get class names for prediction index
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    dnew = xgb.DMatrix(features)
    predictions = loaded_model.predict(dnew)

    top_num = 5
    top5_indices = np.argsort(predictions, axis=1)[:, -top_num:]

    top5_probs = np.array([sample[indices] for sample, indices in zip(predictions, top5_indices)])


    predictions = []
    for i in range(top_num):
        class_index = top5_indices[0, -i]
        predicted_class_name = idx_to_class[class_index]
        predicted_probability = top5_probs[0, -i]
        predictions.append({'Category ID': predicted_class_name, 'Probability': predicted_probability})

    df = pd.DataFrame(predictions)
    df.sort_values(by='Probability', ascending=False, inplace=True)
    df_string = df.to_string(index=False)
    print(df_string)

def image_to_features_resnet(new_image_path):
    weights = 'ResNet50_Weights.DEFAULT'
    resnet = models.resnet50(weights=weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)

    num_classes = 81
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes, device = device)

    file_path = 'resnet50_state_dict.pth'
    folder = 'DL_dicts'
    full_path = os.path.join(folder, file_path)

    resnet.load_state_dict(torch.load(full_path, map_location=device))


    # Load image
    #image_path = 'extra images test/shiba-inu.webp' # test image path
    #image_path = 'extra images test/animal3.webp' # test image path

    image_path = new_image_path # test image path
    image = Image.open(image_path)

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size that model expects
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460], std=[0.2339, 0.2276, 0.2402]),  # Normalize
    ])

    image = transform(image).unsqueeze(0)  # Add a batch dimension
    image = image.to(device)


    feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feature_extractor.eval()  # Set the model to evaluation mode

    final_features = []

    with torch.no_grad():  # No gradient needed for feature extraction
        features = feature_extractor(image)
        features = torch.flatten(features, start_dim=1)  # Flatten the features
        final_features.append(features.cpu().numpy())
        #features = features.cpu().numpy()

    final_features = np.concatenate(final_features, axis=0)
    features = final_features
    
    return features

def image_to_features_vit(new_image_path):

    # Load pre-trained ResNet model
    model = timm.create_model('vit_base_patch32_clip_224.openai_ft_in1k', pretrained=True) # to check others models as well

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    no_of_classes = 81

    num_classes = no_of_classes
    model.head = nn.Linear(model.head.in_features, num_classes, device = device)

    file_path = 'vit_base_patch32_state_dict.pth'
    folder = 'DL_dicts'
    full_path = os.path.join(folder, file_path)

    model.load_state_dict(torch.load(full_path))

    image_path = new_image_path
    image = Image.open(image_path)

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size that model expects
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460], std=[0.2339, 0.2276, 0.2402]),  # Normalize
    ])


    image = transform(image).unsqueeze(0)  # Add a batch dimension
    image = image.to(device)

    with torch.no_grad():  # No gradient needed for feature extraction
        features = model.forward_features(image)
        features = features.cpu().numpy()

    features = features.mean(axis=1)  # Shape: (1, 768)

    return features