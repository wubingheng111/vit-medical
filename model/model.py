import os
import torch
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    ViTImageProcessor,
    Trainer
)
from datasets import load_from_disk
from PIL import Image
import numpy as np


# Load the data
train_dataset = load_from_disk('./dataset/train')
test_dataset = load_from_disk('./dataset/test')

# Load the image processor
image_processor = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)
num_labels = len(train_dataset.features['label'].names)

# Load the model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_labels, 
    cache_dir='./cache')

def transorm_function(data):
    if not isinstance(data['image'], Image.Image):
        data['image'] = Image.fromarray(data['image'])
    
    if data['image'].mode != 'RGB':
        data['image'] = data['image'].convert('RGB')
    
    image_arry = np.array(data['image'])
    inputs = image_processor(images=image_arry, return_tensors='pt')

    data['pixel_values'] = inputs.pixel_values[0]
    return data


train_dataset = train_dataset.map(transorm_function)
test_dataset = test_dataset.map(transorm_function)

# Define the training arguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50, 
    logging_dir='./logs',
    report_to='wandb',
    run_name='vit-medical-image-classification'
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model('./model')