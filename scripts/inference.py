import torch
from transformers import AutoModelForImageClassification, AutoTokenizer
from torchvision import transforms
from PIL import Image
import os

def load_model(model_name):
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.logits, 1)
    return predicted.item()

def main(image_path, model_name):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(model_name)
    image_tensor = preprocess_image(image_path)
    prediction = predict(model, image_tensor)

    print(f"Predicted class index: {prediction}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script for ViT model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    args = parser.parse_args()

    main(args.image, args.model)