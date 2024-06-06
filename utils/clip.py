import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)
import json
import clip


def test_clip(class_to_idx_path, test_loader):
    # Load class names
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # Get and preprocess class names into a format suitable for CLIP
    categories = list(class_to_idx.keys())
    category_texts = clip.tokenize(categories).to(device)
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device)
    model = model.to(device)
    
    # Define the evaluation function
    def evaluate_model_clip(model, data_loader, categories, category_texts):
        model.eval()  # Set the model to evaluation mode

        true_labels = []
        predicted_labels = []

        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in tqdm(data_loader, desc="Testing: "):
                # Preprocess images
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Encode the images and the category texts
                image_features = model.encode_image(inputs)
                text_features = model.encode_text(category_texts)

                # Compute the similarity between the image and each category
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                # Get the top category
                _, preds = torch.max(similarities, 1)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        return accuracy, precision, recall, f1

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model_clip(model, test_loader, categories, category_texts)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics_dict
