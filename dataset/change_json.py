import json


json_file_path = "/linting-slow-vol/Visual-RFT/dataset/oric_train_data.json"
output_path = "/linting-slow-vol/Visual-RFT/dataset/oric_train_data_new.json"

with open(json_file_path, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    image = item['image']
    problem = item['problem']
    # problem = problem.replace('Please answer yes or no based on the given image. ', 'Please answer the question based on the image. ')
    problem = problem.replace('Please answer yes or no based on the given image. ', 'Please think step-by-step and answer the question based on the given image.\n')
    problem = problem.replace('<answer>species name</answer>', '<answer>yes or no</answer>')
    solution = item['solution']
    new_item = {
        'image': image,
        'problem': problem,
        'solution': solution
    }
    new_data.append(new_item)

with open(output_path, 'w') as f:
    json.dump(new_data, f, indent=4)