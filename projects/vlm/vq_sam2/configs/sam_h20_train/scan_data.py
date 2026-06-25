import os
import json
import tqdm

sam_folder = "/mnt/hdfs/byte_ttlive_strategy/zhangtao/sam_datas/"

ret = []
print("Listing the folder !!!")
files = os.listdir(sam_folder)
print("Finish Listing the folder !!!")
for file_name in tqdm.tqdm(files):
    if ".json" not in file_name:
        continue
    ret.append({'image_file': file_name.replace(".json", ".jpg"), 'json_file': file_name})

with open("/mnt/hdfs/byte_ttlive_strategy/zhangtao/sam_infos.json", "w") as f:
    json.dump(ret, f)