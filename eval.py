from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    ViTImageProcessor,
    Trainer
)
from transformers.data.data_collator import DefaultDataCollator
# 导入image_processor
from transformers import ViTImageProcessorFast
from datasets import load_from_disk
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score


test_dataset = load_from_disk('./dataset/test')

image_processor = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)

def transorm_function(data):
    if not isinstance(data['image'], Image.Image):
        data['image'] = Image.fromarray(data['image'])
    
    if data['image'].mode != 'RGB':
        data['image'] = data['image'].convert('RGB')
    
    image_arry = np.array(data['image'])
    inputs = image_processor(images=image_arry, return_tensors='pt')
    data['pixel_values'] = inputs.pixel_values[0]
    return data


test_dataset = test_dataset.map(transorm_function)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


model = ViTForImageClassification.from_pretrained('./results/checkpoint-2256')

training_args = TrainingArguments(
    output_dir='./results_evaluate',
    per_device_eval_batch_size=8,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)