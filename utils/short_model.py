import json
from utils.resnet_vit_download import get_vit_file, get_rn_file
import matplotlib.pyplot as plt
import pandas as pd
import os


def get_categories_vit(new_image_path):

    import pandas as pd
    from PIL import Image
    import timm
    import torch
    from torchvision import transforms
    import torch.nn as nn
    import torch.nn.functional as F

    # не используется
    # Определение трансформаций
    # transform_norm_new = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     # Наши подсчитанные значения mean и std для нормализации
    #     transforms.Normalize(mean=[0.5018, 0.4925, 0.4460],
    #                          std=[0.2339, 0.2276, 0.2402])
    # ])

    # Load pre-trained VIT model
    model = timm.create_model('vit_base_patch32_clip_224.openai_ft_in1k',
                              pretrained=True)  # check others models as well

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    num_classes = 81  # no_of_classes
    model.head = nn.Linear(model.head.in_features,
                           num_classes, device=device)

    # Then load the state dictionary
    get_vit_file()  # download file if not exists

    dl_folder = 'DL_dicts'
    file_path = 'vit_base_patch32_state_dict.pth'
    full_path = os.path.join(dl_folder, file_path)

    model.load_state_dict(torch.load(full_path, map_location=device))

    # Test the model on one image
    # Load image
    # image_path='extra images test/shiba-inu.webp' # test image path
    image_path = new_image_path  # test image path
    image = Image.open(image_path)

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size that model expects
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460],
                             std=[0.2339, 0.2276, 0.2402]),  # Normalize
    ])

    image = transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():  # Disable gradient tracking
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)

    # Get top categories
    top_num = 5  # Number of top categories you want
    top_prob, top_catid = torch.topk(probabilities, top_num)

    # Load category names
    with open('class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)

    # Get class names for prediction index
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert to Python data types and print
    top_prob = top_prob.cpu().numpy()[0]
    top_catid = top_catid.cpu().numpy()[0]

    predictions = []
    for i in range(top_num):
        predicted_class_name = idx_to_class[top_catid[i]]
        predicted_probability = top_prob[i]
        predictions.append({'Category ID': predicted_class_name,
                            'Probability': predicted_probability})

    df = pd.DataFrame(predictions)
    df_string = df.to_string(index=False)
    print(df_string)

    return df_string  # return df


def get_categories_rn(new_image_path):
    import pandas as pd
    from PIL import Image

    import torch
    from torchvision import transforms
    import torchvision.models as models
    import torch.nn as nn
    import torch.nn.functional as F

    # Load image
    image_path = new_image_path  # test image path
    image = Image.open(image_path)

    # Load image
    # image_path = 'extra images test/horse.jpg' # test image path
    # image = Image.open(image_path)

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size your model expects
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460],
                             std=[0.2339, 0.2276, 0.2402]),  # Normalize
    ])

    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Load category names
    with open('class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)

    # First, recreate the model architecture
    weights = 'ResNet50_Weights.DEFAULT'
    resnet = models.resnet50(weights=weights)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    resnet = resnet.to(device)

    # num_classes = len(class_to_idx)
    num_classes = 81
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes, device=device)

    # Then load the state dictionary
    get_rn_file()  # download file if not exists

    dl_folder = 'DL_dicts'
    file_path = 'resnet50_state_dict.pth'
    full_path = os.path.join(dl_folder, file_path)

    resnet.load_state_dict(torch.load(full_path, map_location=device))

    # Move model to GPU
    resnet = resnet.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model to evaluation mode
    resnet.eval()

    with torch.no_grad():  # Disable gradient tracking
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = resnet(image)
        probabilities = F.softmax(outputs, dim=1)

    # Get top categories
    top_num = 5
    top_prob, top_catid = torch.topk(probabilities, top_num)

    # Get class names for prediction index
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert to Python data types and print
    top_prob = top_prob.cpu().numpy()[0]
    top_catid = top_catid.cpu().numpy()[0]

    predictions = []
    for i in range(top_num):
        predicted_class_name = idx_to_class[top_catid[i]]
        predicted_probability = top_prob[i]
        predictions.append({'Category ID': predicted_class_name,
                            'Probability': predicted_probability})

    df = pd.DataFrame(predictions)
    df_string = df.to_string(index=False)
    print(df_string)

    return df_string  # return df


def get_categories_clip(new_image_path):
    import pandas as pd
    from PIL import Image
    import clip
    import json

    import torch
    from torchvision import transforms
    import torchvision.models as models
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device)
    
    # Load cat names
    with open('class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)

    # Get and preprocess cat names into a format suitable for CLIP
    cat_names = list(class_to_idx.keys())
    categories = cat_names    
    category_texts = clip.tokenize(categories).to(device)

    # Load image
    image_path = new_image_path  # test image path
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Encode the image and the category texts
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(category_texts)

    # Compute the similarity between the image and each category
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Get the top 5 categories and their probabilities
    top_num = 5
    top_prob, top_catid = torch.topk(similarities, top_num)

    # Convert to Python data types
    top_prob = top_prob.cpu().numpy()[0]
    top_catid = top_catid.cpu().numpy()[0]

    # Map indices to class names and prepare predictions
    predictions = []
    for i in range(top_num):
        predicted_class_name = categories[top_catid[i]]
        predicted_probability = top_prob[i]
        predictions.append({'Category ID': predicted_class_name,
                            'Probability': predicted_probability})

    df = pd.DataFrame(predictions)
    df_string = df.to_string(index=False)
    print(df_string)

    return df_string  # return df


def save_result_as_chart(result):
    df = pd.DataFrame(result)
    dpi = 60

    plt.figure(figsize=(1.5, 1.5))

    # Plot the bar chart
    ax = df.plot(kind='bar', x='Category ID', y='Probability', color='red')

    # Manually set the y-axis range
    ax.set_ylim([0, 1.1])

    # Add number labels to each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',  # 2 decimals
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=14)

    # Set title and adjust the font size
    plt.title('Top categories', fontsize=20)

    # Adjust font size for labels and ticks
    plt.xlabel('Category ID', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Adjust plot margins
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)

    filename = 'result.png'
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)

    return filename
