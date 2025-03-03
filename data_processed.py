from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets, DatasetDict

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)

size = (480, 480)
# resize to 480x480
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["images"]]
    del examples["images"]
    return examples


if __name__ == "__main__":
    for i in range(10):
        dataset = load_from_disk("./datasets")
        dataset = concatenate_datasets(
            [dataset['train'], dataset['test']]
        )
        dataset_len = len(dataset) // 10
        dataset = dataset.select(range(i * dataset_len, (i + 1) * dataset_len))
        dataset = dataset.map(transforms, batched=True, num_proc=1)
        dataset.set_format(type="torch", columns=["pixel_values", "labels"])
        dataset.save_to_disk(f"./split_datasets_{i}")


    # concatenate datasets
    dataset = concatenate_datasets(
        [load_from_disk(f"./split_datasets_{i}") for i in range(10)]
    )

    dataset.save_to_disk("./split_datasets")


    dataset = dataset.map(transforms, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    dataset.save_to_disk("./split_datasets")
    split_datasets = load_from_disk("./split_datasets")
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']

    for i, example in enumerate(train_dataset):
        if example['labels'] == None or example['pixel_values'] == None:
            print(i)
            print(example)

