from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

import torchvision.models as models
import torch.nn as nn
from torch.optim import Adam

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle


def prepare_data_resnet(images_folder_path):
    
    # Load dataset
    #dataset_path = 'Test1_large/'
    #dataset_path = 'Data_small/'
    dataset_path = images_folder_path
    
    # Определение трансформаций 
    transform_norm = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460], # Наши подсчитанные значения mean и std для нормализации
                            std=[0.2339, 0.2276, 0.2402])
    ])

    dataset_norm = ImageFolder(dataset_path, transform=transform_norm)


    
    

    image, label = dataset_norm[0]

    image = image.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    mean=[0.5018, 0.4925, 0.4460]
    std=[0.2339, 0.2276, 0.2402]
    image = image * std + mean  # Denormalize
    image = np.clip(image, 0, 1)  # Clip values to be between 0 and 1

    no_of_classes = len(dataset_norm.classes)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Splitting dataset into Train and Test

    labels = dataset_norm.classes

    

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    # Convert class weights to a tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Split the dataset
    

    # Define sizes for train and test sets
    total_size = len(dataset_norm)
    test_size = int(0.2 * total_size)  # 20% for testing
    train_size = total_size - test_size  # Remaining 80% for training

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset_norm, [train_size, test_size])

    

    # Extract labels for the training dataset
    train_labels = [label for _, label in train_dataset]

    # Create a list of sample weights based on class weights
    train_sample_weights = [class_weights_tensor[label].item() for label in train_labels]

    # Create the sampler
    train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

    num_workers = max(0, os.cpu_count() or 0)
    print(f'Using {num_workers} workers for data loading')

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = num_workers)
    dataloader = DataLoader(dataset_norm, batch_size=32, shuffle=True, num_workers=num_workers)

    return device, no_of_classes, train_loader, test_loader, dataloader

def train_resnet(device, no_of_classes, train_loader, test_loader):
    # DO A FINE-TUNED RESNET

    # Load pre-trained ResNet model
    #weights='ResNet50_Weights.IMAGENET1K_V1'
    weights = 'ResNet50_Weights.DEFAULT'
    
    #resnet = models.resnet50(weights=True)
    resnet = models.resnet50(weights=weights)
    resnet = resnet.to(device)

    # Freeze all layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer to match number of classes
    num_classes = no_of_classes
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes, device = device)

    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(resnet.fc.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10 


    for epoch in tqdm(range(num_epochs), desc="Optimizing: "):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) # move to GPU if available
            
            optimizer.zero_grad() # Zero the parameter gradients
            
            # Forward pass
            outputs = resnet(inputs).to(device) # move to GPU if available
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    

    # Save the model's state dictionary
    folder = 'DL_dicts'
    file = 'resnet50_state_dict.pth'
    full_path = os.path.join(folder, file)

    torch.save(resnet.state_dict(), full_path)

    return no_of_classes, test_loader

def test_resnet(no_of_classes, test_loader):
    # First, recreate the model architecture
    weights = 'ResNet50_Weights.DEFAULT'
    
    #resnet = models.resnet50(weights=True)
    resnet = models.resnet50(weights=weights)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    resnet = resnet.to(device)

    num_classes = no_of_classes
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes, device = device)

    # Then load the state dictionary
    folder = 'DL_dicts'
    file = 'resnet50_state_dict.pth'
    full_path = os.path.join(folder, file)

    resnet.load_state_dict(torch.load(full_path, map_location=device))

    # TESTING


    def evaluate_model(model, data_loader):
        model.eval()  # Set the model to evaluation mode

        true_labels = []
        predicted_labels = []

        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in tqdm(data_loader, desc="Testing: "):
                inputs, labels = inputs.to(device), labels.to(device) # move to GPU if available
                outputs = model(inputs).to(device) # move to GPU if available
                _, preds = torch.max(outputs, 1)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        return accuracy, precision, recall, f1
    
    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(resnet, test_loader)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy
    metrics_dict['precision'] = precision
    metrics_dict['recall'] = recall
    metrics_dict['f1'] = f1
    
    return metrics_dict


def extract_features_resnet(device, no_of_classes, dataloader):
    # Load RESNET weights and extract features

    weights = 'ResNet50_Weights.DEFAULT'
    

    resnet = models.resnet50(weights=weights)
    resnet = resnet.to(device)

    num_classes = no_of_classes 
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)

    # Load the state dictionary
    folder = 'DL_dicts'
    file = 'resnet50_state_dict.pth'
    full_path = os.path.join(folder, file)

    resnet.load_state_dict(torch.load(full_path, map_location=device))

    feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)

    # Feature extraction process

    final_features = []
    final_labels = []

    for inputs, labels in tqdm(dataloader, desc="Extracting Features: "):
        inputs = inputs.to(device)
        
        with torch.no_grad():  # No gradient needed for feature extraction
            # Extract features
            features = feature_extractor(inputs)
            features = torch.flatten(features, start_dim=1)  # Flatten the features
            final_features.append(features.cpu().numpy())
            final_labels.append(labels.numpy())

    final_features = np.concatenate(final_features, axis=0)
    final_labels = np.concatenate(final_labels, axis=0)

    # Convert to DataFrame and save to CSV
    file_name = 'resnet_features.csv'
    feature_folder = 'features'

    # Create the folder if it doesn't already exist
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    
    file_path = os.path.join(feature_folder, file_name)

    df_features = pd.DataFrame(final_features)
    df_features['label'] = final_labels
    df_features.to_csv(file_path, index=False)

    return final_features, final_labels, file_path

