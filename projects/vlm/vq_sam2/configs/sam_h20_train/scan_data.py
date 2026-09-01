import os
import json
import tqdm

sam_folder = os.environ.get("SAM_DATA_ROOT", "data/sam")
sam_info_json = os.environ.get("SAM_INFO_JSON", "data/sam_infos.json")

ret = []
print("Listing the folder !!!")
files = os.listdir(sam_folder)
print("Finish Listing the folder !!!")
for file_name in tqdm.tqdm(files):
    if ".json" not in file_name:
        continue
    ret.append({'image_file': file_name.replace(".json", ".jpg"), 'json_file': file_name})

with open(sam_info_json, "w") as f:
    json.dump(ret, f)
