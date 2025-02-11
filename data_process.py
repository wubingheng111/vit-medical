import os
import zipfile
from datasets import load_dataset


# unzip the data
def unzip_data(src_path,target_path):
    if(not os.path.isdir(target_path + "train")):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()

# unzip_data("./Data.zip","./dataset/")

# load the data
image_dataset_train = load_dataset('imagefolder',data_dir='./data/train', split='train')
image_dataset_train.save_to_disk('./dataset/train')

image_dataset_test = load_dataset('imagefolder',data_dir='./data/test', split='train')
image_dataset_test.save_to_disk('./dataset/test')




