import re


MASK_GENERATION_REFSEG = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_refseg713k.json",
    "data_path": "",
}

MASK_GENERATION_INVIG = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_invig505k.json",
    "data_path": "",
}

MASK_GENERATION_GCG = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_gcg196k.json",
    "data_path": "",
}

MASK_GENERATION_REASONSEG = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg61k.json",
    "data_path": "",
}

MASK_GENERATION_V3DET = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_v3det183k.json",
    "data_path": "",
}

MASK_UNDERSTANDING_DAM = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_understanding_dam1458k.json",
    "data_path": "",
}

MASK_GENERATION_DW = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_denseworld900k.json",
    "data_path": "",
}

MASK_GENERATION_COCONUT = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_coconut426k_sft_298k.json",
    "data_path": "",
}


data_dict = {
    "mask_generation_refseg": MASK_GENERATION_REFSEG,
    "mask_generation_invig": MASK_GENERATION_INVIG,
    "mask_generation_gcg": MASK_GENERATION_GCG,
    "mask_generation_reasonseg": MASK_GENERATION_REASONSEG,
    "mask_generation_v3det": MASK_GENERATION_V3DET,
    "mask_understanding_dam": MASK_UNDERSTANDING_DAM,
    "mask_generation_dw": MASK_GENERATION_DW,
    "mask_generation_coconut": MASK_GENERATION_COCONUT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

if __name__ == "__main__":
    dataset_names = ["llava_665k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)