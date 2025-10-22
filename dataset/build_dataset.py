from datasets import load_dataset, DatasetDict

file_path = "oric_train_data.json"

dataset = load_dataset("json", data_files=file_path)

oric = DatasetDict({
    "train": dataset["train"]
})

save_path = "/ariesdv0/zhaoyang/dataset/oric_train_data_dataset"

oric.save_to_disk(save_path)
print(f"Dataset saved to {save_path}")