from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import os

from tqdm import tqdm
import torch.nn as nn
from timm.optim import AdamP
import timm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import pandas as pd



def prepare_data_vit(images_folder_path):


    # Load dataset
    #dataset_path = 'Test1_large/'
    #dataset_path = 'Data_small/'
    dataset_path = images_folder_path

    # Определение трансформаций 
    transform_norm_new = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460], # Наши подсчитанные значения mean и std для нормализации
                            std=[0.2339, 0.2276, 0.2402])
    ])

    dataset_norm = ImageFolder(dataset_path, transform=transform_norm_new)
    no_of_classes = len(dataset_norm.classes)


    # Split the dataset
    from torch.utils.data import random_split

    # Define sizes for train and test sets
    total_size = len(dataset_norm)
    test_size = int(0.2 * total_size)  # 20% for testing
    train_size = total_size - test_size  # Remaining 80% for training

    # Split the dataset
    train_dataset_adj, test_dataset_adj = random_split(dataset_norm, [train_size, test_size])

    # Save dataset_norm class names
    class_to_idx = dataset_norm.class_to_idx

    with open('class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f)
    
    num_workers = max(0, os.cpu_count() or 0)
    print(f'Using {num_workers} workers for data loading')

    train_loader_adj = DataLoader(train_dataset_adj, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader_adj = DataLoader(test_dataset_adj, batch_size=32, shuffle=False, num_workers=num_workers)
    dataloader = DataLoader(dataset_norm, batch_size=32, shuffle=True, num_workers=num_workers)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    return device, no_of_classes, train_loader_adj, test_loader_adj, dataloader
        
def train_vit(device, no_of_classes, train_loader, test_loader):
    import timm
    import torch.nn as nn
    from torch.optim import Adam

    #model_names = timm.list_models('vit*', pretrained=True)
    #print(model_names)


    # Load pre-trained ResNet model
    model = timm.create_model('vit_base_patch32_clip_224.openai_ft_in1k', pretrained=True) # to check other models as well

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)


    # Modify the final fully connected layer to match  number of classes
    num_classes = no_of_classes
    model.head = nn.Linear(model.head.in_features, num_classes, device = device)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Only train the last layer
    for param in model.head.parameters():
        param.requires_grad = True
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.head.parameters(), lr=0.001) # try AdamW

    # Training loop
    num_epochs = 10  # or any number of epochs you prefer

    torch.set_grad_enabled(True) # testing!!!

    for epoch in tqdm(range(num_epochs), desc="Optimizing: "):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) # move to GPU if available

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).to(device) # move to GPU if available
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


    # Save the model's state dictionary
    folder = 'DL_dicts'
    file = 'vit_base_patch32_state_dict.pth'
    full_path = os.path.join(folder, file)

    torch.save(model.state_dict(), full_path)

    return no_of_classes, test_loader

def test_vit(no_of_classes, test_loader):
    model = timm.create_model('vit_base_patch32_clip_224.openai_ft_in1k', pretrained=True) # to check others models as well

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    num_classes = no_of_classes
    model.head = nn.Linear(model.head.in_features, num_classes, device = device)

    # Then load the state dictionary
    folder = 'DL_dicts'
    file = 'vit_base_patch32_state_dict.pth'
    full_path = os.path.join(folder, file)
    model.load_state_dict(torch.load(full_path))


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
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

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


def extract_features_vit(device, no_of_classes, dataloader):
    # Load pre-trained VIT model
    model = timm.create_model('vit_base_patch32_clip_224.openai_ft_in1k', pretrained=True) # to check others models as well

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_classes = no_of_classes
    model.head = nn.Linear(model.head.in_features, num_classes, device = device)

    # Then load the state dictionary
    folder = 'DL_dicts'
    file = 'vit_base_patch32_state_dict.pth'
    full_path = os.path.join(folder, file)
    model.load_state_dict(torch.load(full_path))
   

    # Extract features
    final_features = []
    final_labels = []
    with torch.no_grad():  # No gradient needed for feature extraction
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model.forward_features(inputs)
            final_features.append(features.cpu().numpy())
            final_labels.append(labels.numpy())

    final_features = np.concatenate(final_features, axis=0)
    final_labels = np.concatenate(final_labels, axis=0)

    # Convert to DataFrame and save to CSV
    file_name = 'vit_features.csv'
    feature_folder = 'features'

    # Create the folder if it doesn't already exist
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    
    file_path = os.path.join(feature_folder, file_name)

    df_features = pd.DataFrame(final_features)
    df_features['label'] = final_labels
    df_features.to_csv(file_path, index=False)

    return final_features, final_labels, file_path