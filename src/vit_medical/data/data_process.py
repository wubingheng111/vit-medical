from datasets import load_dataset, load_from_disk, Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import pickle
import numpy as np

# def get_name2label(path):
#     name2label = {}
#     for name in os.listdir(path):
#         if os.path.isdir(os.path.join(path, name)):
#             name2label[name] = len(name2label)
#     return name2label

# name2label = get_name2label('./origin-data')
# label2name = {v: k for k, v in name2label.items()}
# print(name2label)
# print(label2name)

# with open('./tokenizer/nametolabel.pkl', 'wb') as f:
#     pickle.dump(name2label, f)
# with open('./tokenizer/nametolabel.pkl', 'wb') as f:
#     pickle.dump(label2name, f)


def build_dataset_dict(path, name2label):
    image_paths = []
    labels = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            label = name2label[name]
            for img_name in os.listdir(os.path.join(path, name)):
                img_path = os.path.join(path, name, img_name)
                image_paths.append(img_path)
                labels.append(label)
    return {'image_path': image_paths, 'labels': labels}


def process_image(example):
    img = Image.open(example['image_path'])
    return {'images': img, 'labels': example['labels']}

# with open('./tokenizer/nametolabel.pkl', 'wb') as f:
#     pickle.dump(name2label, f)
# with open('./tokenizer/labeltoname.pkl', 'wb') as f:  # 使用不同的文件名
#     pickle.dump(label2name, f)

# load tokenizer
with open('./tokenizer/nametolabel.pkl', 'rb') as f:
    name2label = pickle.load(f)

data_dict = build_dataset_dict('./origin-data', name2label)
dataset = Dataset.from_dict(data_dict)


if __name__ == "__main__":
    processed_dataset = dataset.map(
        process_image,
        remove_columns=['image_path'],
        batched=False,
        num_proc=4
    )
    print(processed_dataset)
    print(processed_dataset[0]["images"])
    print(processed_dataset[0]["labels"])

    processed_dataset.save_to_disk('./processed_dataset')

# from datasets import Dataset
# from datasets import load_from_disk

# datasets = load_from_disk('./processed_dataset')
# print(datasets)
# split_datasets = datasets.train_test_split(test_size=0.2)
# print(split_datasets)
# split_datasets.save_to_disk('./datasets')