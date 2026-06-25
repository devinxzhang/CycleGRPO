import os
import json
import tqdm

def scandir_generator(path):
    with os.scandir(path) as entries:
        for entry in entries:
            yield entry.name  # 逐个返回文件名，不占用大量内存

# coconut_list = []
# for json_file in scandir_generator("temp_data/coconut"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("temp_data/coconut", json_file), 'r') as f:
#         print("==========>>>", os.path.join("temp_data/coconut", json_file))
#         try:
#             data_dict = json.load(f)
#         except:
#             continue
#     coconut_list.append(data_dict)


# num_samples = len(coconut_list) // 1000

# with open(f"data/mask_generation_coconut_insseg{num_samples}k.json", 'w') as f:
#     json.dump(coconut_list, f)


# denseworld_list = []
# for json_file in scandir_generator("temp_data_256x2_0927/denseworld"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("temp_data_256x2_0927/denseworld", json_file), 'r') as f:
#         print("==========>>>", os.path.join("temp_data_256x2_0927/denseworld", json_file))
#         try:
#             data_dict = json.load(f)
#         except:
#             continue
#     denseworld_list.extend(data_dict)

# num_samples = len(denseworld_list) // 1000

# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_denseworld{num_samples}k.json", 'w') as f:
#     json.dump(denseworld_list, f)


# v3det_list = []
# for json_file in scandir_generator("temp_data/v3det_grounding"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("temp_data/v3det_grounding", json_file), 'r') as f:
#         print("==========>>>", os.path.join("temp_data/v3det_grounding", json_file))
#         try:
#             data_dict = json.load(f)
#         except:
#             continue
#     conversations = data_dict['conversations']
#     question, answer = conversations[-2:]
#     new_conversation = [
#         {'from': 'human', 'value': "<image>\n"+question['value']},
#         answer
#     ]

#     new_data_dict = {
#         'image': data_dict['image'],
#         'conversations': new_conversation,
#     }
#     v3det_list.append(new_data_dict)

# num_samples = len(v3det_list) // 1000
# with open(f"data/mask_generation_v3det_insseg{num_samples}k.json", 'w') as f:
#     json.dump(v3det_list, f)


# sa1b_box2mask_mask2box = []
# for json_file in scandir_generator("temp_data/sa1b_box2mask_mask2box"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("temp_data/sa1b_box2mask_mask2box", json_file), 'r') as f:
#         print("==========>>>", os.path.join("temp_data/sa1b_box2mask_mask2box", json_file))
#         data_dict = json.load(f)
#     sa1b_box2mask_mask2box.append(data_dict)

# num_samples = len(sa1b_box2mask_mask2box) // 1000

# with open(f"data/mask_box_alignment_sa1b{num_samples}k.json", 'w') as f:
#     json.dump(sa1b_box2mask_mask2box, f)


# dam_list = []
# for json_file in scandir_generator("temp_data_256x2_0927/dam"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("temp_data_256x2_0927/dam", json_file), 'r') as f:
#         print("==========>>>", os.path.join("temp_data_256x2_0927/dam", json_file))
#         data_dict = json.load(f)
#     dam_list.extend(data_dict)

# num_samples = len(dam_list) // 1000

# with open(f"data/vq_sam2_data_256x2_0927/mask_understanding_dam{num_samples}k.json", 'w') as f:
#     json.dump(dam_list, f)


# refseg_list = []

# for split_name in scandir_generator("./temp_data_256x2_1005/ref_seg"):
#     if split_name == 'invig500k':
#         continue
#     for json_file in os.listdir(os.path.join("./temp_data_256x2_1005/ref_seg", split_name)):
#         if not json_file.endswith(".json"):
#             print("file not found.........")
#             continue
#         with open(os.path.join("./temp_data_256x2_1005/ref_seg", split_name, json_file), 'r') as f:
#             print("==========>>>", os.path.join("./temp_data_256x2_1005/ref_seg", split_name, json_file))
#             data_dict = json.load(f)
#         refseg_list.extend(data_dict)

# num_samples = len(refseg_list) // 1000

# with open(f"data/vq_sam2_data_256x2_1005/mask_generation_refcoco{num_samples}k.json", 'w') as f:
#     json.dump(refseg_list, f)


# refseg_list = []

# for split_name in scandir_generator("./temp_data_256x2_0927/ref_seg"):
#     if split_name == 'invig500k' or split_name == 'refcoco' or split_name == 'refcoco+' or split_name == 'refcocog':
#         continue
#     for json_file in os.listdir(os.path.join("./temp_data_256x2_0927/ref_seg", split_name)):
#         if not json_file.endswith(".json"):
#             print("file not found.........")
#             continue
#         with open(os.path.join("./temp_data_256x2_0927/ref_seg", split_name, json_file), 'r') as f:
#             print("==========>>>", os.path.join("./temp_data_256x2_0927/ref_seg", split_name, json_file))
#             data_dict = json.load(f)
#         refseg_list.extend(data_dict)

# num_samples = len(refseg_list) // 1000

# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_refseg_exclude_refcoco{num_samples}k.json", 'w') as f:
#     json.dump(refseg_list, f)



# padt_refseg_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/padt_refcoco"):
#     if not json_file.endswith(".json"):
#         print("file not found.........")
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/padt_refcoco", json_file), 'r') as f:
#         data_dict = json.load(f)
#         padt_refseg_list.extend(data_dict)
# num_samples = len(padt_refseg_list) // 1000

# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_padt_refcoco{num_samples}k.json", 'w') as f:
#     json.dump(padt_refseg_list, f)


# padt_ric_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/padt_ric"):
#     if not json_file.endswith(".json"):
#         print("file not found.........")
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/padt_ric", json_file), 'r') as f:
#         data_dict = json.load(f)
#         padt_ric_list.extend(data_dict)
# num_samples = len(padt_ric_list) // 1000

# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_padt_ric{num_samples}k.json", 'w') as f:
#     json.dump(padt_ric_list, f)


# padt_cocoseg_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/padt_cocodet"):
#     if not json_file.endswith(".json"):
#         print("file not found.........")
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/padt_cocodet", json_file), 'r') as f:
#         data_dict = json.load(f)
#         padt_cocoseg_list.extend(data_dict)
# num_samples = len(padt_cocoseg_list) // 1000

# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_padt_cocoseg{num_samples}k.json", 'w') as f:
#     json.dump(padt_cocoseg_list, f)



# gcg_list = []
# for json_file in scandir_generator("./temp_data_256x2_0927/gcg"):
#     if not json_file.endswith(".json"):
#         continue
#     if json_file.startswith('grandf'):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/gcg", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/gcg", json_file))
#         data_dict = json.load(f)
#     gcg_list.extend(data_dict)
# num_samples = len(gcg_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_gcg_exclude_grandf{num_samples}k.json", 'w') as f:
#     json.dump(gcg_list, f)

# gcg_list = []
# for json_file in scandir_generator("./temp_data_256x2_0927/gcg"):
#     if not json_file.endswith(".json"):
#         continue
#     if not json_file.startswith('grandf'):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/gcg", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/gcg", json_file))
#         data_dict = json.load(f)
#     gcg_list.extend(data_dict)
# num_samples = len(gcg_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_gcg_grandf{num_samples}k.json", 'w') as f:
#     json.dump(gcg_list, f)


# reasonseg_list = []
# for split_name in os.listdir("temp_data_256x2_0927/lisa_plus"):
#     for json_file in os.listdir(os.path.join("temp_data_256x2_0927/lisa_plus", split_name)):
#         if not json_file.endswith(".json"):
#             continue
#         with open(os.path.join("temp_data_256x2_0927/lisa_plus", split_name, json_file), 'r') as f:
#             print("==========>>>", os.path.join("temp_data_256x2_0927/lisa_plus", split_name, json_file))
#             data_dict = json.load(f)
#         reasonseg_list.extend(data_dict)
# num_samples = len(reasonseg_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_reasonseg{num_samples}k.json", 'w') as f:
#     json.dump(reasonseg_list, f)


# muse_list = []
# for json_file in scandir_generator("./temp_data_256x2_0927/muse"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/muse", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/muse", json_file))
#         data_dict = json.load(f)
#     muse_list.extend(data_dict)
# num_samples = len(muse_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_muse{num_samples}k.json", 'w') as f:
#     json.dump(muse_list, f)

# muse_list = []
# for json_file in scandir_generator("./temp_data_256x2_0927/segllm"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/segllm", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/segllm", json_file))
#         data_dict = json.load(f)
#     muse_list.extend(data_dict)
# num_samples = len(muse_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_segllm{num_samples}k.json", 'w') as f:
#     json.dump(muse_list, f)


# grefcoco_list = []
# for json_file in scandir_generator("./temp_data_256x2_0927/ref_seg_v2/grefs"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/ref_seg_v2/grefs", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/ref_seg_v2/grefs", json_file))
#         data_dict = json.load(f)
#     grefcoco_list.extend(data_dict)
# num_samples = len(grefcoco_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_grefcoco{num_samples}k.json", 'w') as f:
#     json.dump(grefcoco_list, f)


# split_path = [
#     "<PATH_TO_DATA>/discrete_spatial_tokenizer/temp_data_256x2_0927/ref_seg_v2/refclef",
#     "<PATH_TO_DATA>/discrete_spatial_tokenizer/temp_data_256x2_0927/ref_seg_v2/refgta",
#     "<PATH_TO_DATA>/discrete_spatial_tokenizer/temp_data_256x2_0927/ref_seg/cops_ref",
#     "<PATH_TO_DATA>/discrete_spatial_tokenizer/temp_data_256x2_0927/ref_seg/invig21k",
#     "<PATH_TO_DATA>/discrete_spatial_tokenizer/temp_data_256x2_0927/ref_seg/visual7w",
# ]
# refseg_list = []
# for split_dir in split_path:
#     for json_file in os.listdir(split_dir):
#         if not json_file.endswith(".json"):
#             continue
#         with open(os.path.join(split_dir, json_file), 'r') as f:
#             print("==========>>>", os.path.join(split_dir, json_file))
#             data_dict = json.load(f)
#         refseg_list.extend(data_dict)
# num_samples = len(refseg_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_refseg{num_samples}k.json", 'w') as f:
#     json.dump(refseg_list, f)
# print("Saved at ", f"data/vq_sam2_data_256x2_0927/mask_generation_refseg{num_samples}k.json")

# mmr_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/mmr"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/mmr", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/mmr", json_file))
#         data_dict = json.load(f)
#         mmr_list.extend(data_dict)
# num_samples = len(mmr_list) // 1000
# with open(f"data/vq_sam2_data_256x2_0927/mask_generation_mmr{num_samples}k.json", 'w') as f:
#     json.dump(mmr_list, f)
# print("Saved at ", f"data/vq_sam2_data_256x2_0927/mask_generation_mmr{num_samples}k.json")



# psgv1_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/psg_v1"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/psg_v1", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/psg_v1", json_file))
#         data_dict = json.load(f)
#         psgv1_list.extend(data_dict)
# num_samples = len(psgv1_list) // 1000
# with open(f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k.json", 'w') as f:
#     json.dump(psgv1_list, f)
# print("Saved at ", f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k_v1.json")

# psgv2_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/psg_v2"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/psg_v2", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/psg_v2", json_file))
#         data_dict = json.load(f)
#         psgv2_list.extend(data_dict)
# num_samples = len(psgv2_list) // 1000
# with open(f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k_v2.json", 'w') as f:
#     json.dump(psgv2_list, f)
# print("Saved at ", f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k_v2.json")

# psgv3_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/psg_givenmask"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/psg_givenmask", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/psg_givenmask", json_file))
#         data_dict = json.load(f)
#         psgv3_list.extend(data_dict)
# num_samples = len(psgv3_list) // 1000
# with open(f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k_v3.json", 'w') as f:
#     json.dump(psgv3_list, f)
# print("Saved at ", f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k_v3.json")

# psgv4_list = []
# for json_file in os.listdir("./temp_data_256x2_0927/psg_multiround"):
#     if not json_file.endswith(".json"):
#         continue
#     with open(os.path.join("./temp_data_256x2_0927/psg_multiround", json_file), 'r') as f:
#         print("==========>>>", os.path.join("./temp_data_256x2_0927/psg_multiround", json_file))
#         data_dict = json.load(f)
#         psgv4_list.extend(data_dict)
# num_samples = len(psgv4_list) // 1000
# with open(f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k_v4.json", 'w') as f:
#     json.dump(psgv4_list, f)
# print("Saved at ", f"data/tokenmask_data_256x2/mask_generation_psg{num_samples}k_v4.json")


gar_relation_list = []
for json_file in os.listdir("./temp_data_256x2_0927/gar_multi_region_zoom_in"):
    if not json_file.endswith(".json"):
        continue
    with open(os.path.join("./temp_data_256x2_0927/gar_multi_region_zoom_in", json_file), 'r') as f:
        print("==========>>>", os.path.join("./temp_data_256x2_0927/gar_multi_region_zoom_in", json_file))
        data_dict = json.load(f)
        gar_relation_list.extend(data_dict)
num_samples = len(gar_relation_list) // 1000
with open(f"data/tokenmask_data_256x2/mask_understanding_gar_relation{num_samples}k.json", 'w') as f:
    json.dump(gar_relation_list, f, indent=4)
