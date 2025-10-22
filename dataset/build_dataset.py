# from datasets import load_dataset, DatasetDict

# file_path = "oric_train_data.json"

# dataset = load_dataset("json", data_files=file_path)

# oric = DatasetDict({
#     "train": dataset["train"]
# })

# save_path = "/ariesdv0/zhaoyang/dataset/oric_train_data_dataset"

# oric.save_to_disk(save_path)
# print(f"Dataset saved to {save_path}")


import time
from datasets import DatasetDict, Dataset
from PIL import Image
import json
import os

"""
turn your json to DatasetDict
"""
def json_to_dataset(json_file_path, image_root):
    # read json file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    image_paths = [os.path.join(image_root, item['image']) for item in data]
    
    problems = [item['problem'] for item in data]
    solutions = [item['solution'] for item in data]

    images = [Image.open(image_path).convert('RGBA') for image_path in image_paths]

    # dataset_dict = {
    #     'image': images,
    #     'problem': problems,
    #     'solution': solutions
    # }

    dataset_dict = {
        'image': [[p] for p in image_paths],   
        'problem': problems,                   
        'solution': solutions
    }


    dataset = Dataset.from_dict(dataset_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict


"""
save to your local disk
"""
def save_dataset(dataset_dict, save_path):
    # save DatasetDict to your disk
    dataset_dict.save_to_disk(save_path)


"""
read from your local disk
"""
def load_dataset(save_path):
    # load DatasetDict
    return DatasetDict.load_from_disk(save_path)


time1 = time.asctime()
print(time1)
# ### Your dataset in JSON file format consists of three parts: image, problem and solution]
# /ariesdv0/zhaoyang/dataset/coco/resized-2014-train/image-size-384
dataset_dict = json_to_dataset('/ariesdv0/zhaoyang/Visual-RFT/dataset/oric_train_data.json', "/ariesdv0/zhaoyang/dataset/coco/resized-2014-train/image-size-384")
time2 = time.asctime()
save_path = '/ariesdv0/zhaoyang/dataset/oric_train_data_dataset'
save_dataset(dataset_dict, save_path)
print(time2)
test_dataset_dict = load_dataset(save_path)
print(test_dataset_dict)