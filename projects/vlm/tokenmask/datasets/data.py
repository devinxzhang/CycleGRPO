import re



MASK_GENERATION_REFSEG = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_refseg641k.json",
    "data_path": "",
}

MASK_GENERATION_INVIG = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_invig505k.json",
    "data_path": "",
}

MASK_GENERATION_GCG = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_gcg_exclude_grandf195k.json",
    "data_path": "",
}

MASK_GENERATION_GRANDF = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_gcg_grandf1k.json",
    "data_path": "",
}

MASK_GENERATION_REASONSEG = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg61k.json",
    "data_path": "",
}

MASK_GENERATION_V3DET = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_v3det157k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_UNDERSTANDING_DAM = {
    "annotation_path": "./data/tokenmask_data_256x2_cot_format/mask_understanding_dam1458k.json",
    "data_path": "",
}

MASK_GENERATION_DW = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_denseworld872k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_GENERATION_COCONUT = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_coconut422k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_GENERATION_PADT_REFCOCO = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_padt_refcoco321k.json",
    "data_path": "",
}

MASK_GENERATION_PADT_RIC = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_padt_ric561k.json",
    "data_path": "",
}

MASK_GENERATION_PADT_COCOSEG = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_padt_cocoseg113k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_GENERATION_SEGLLM = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_segllm1049k.json",
    "data_path": "",
}

MASK_GENERATION_GRES = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_grefcoco209k_new.json",
    "data_path": "",
}

MASK_GENERATION_LISA = {
    "annotation_path": "./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg1326.json",
    "data_path": "",
}

MASK_GENERATION_PSG_V1 = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_psg44k_v1_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_GENERATION_PSG_V2 = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_psg44k_v2_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_GENERATION_PSG_V3 = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_psg43k_v3_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_GENERATION_PSG_V4 = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_psg43k_v4_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_GENERATION_DW_GCG = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_coconut_gcg128k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_UNDERSTANDING_DAM_ZOOM_IN = {
    "annotation_path": "./data/tokenmask_data_256x2_cot_format/mask_understanding_dam_zoom_in_1168k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_UNDERSTANDING_GAR_ZOOM_IN = {
    "annotation_path": "./data/tokenmask_data_256x2_cot_format/mask_understanding_gar_zoom_in_453k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_UNDERSTANDING_SAM_ZOOM_IN = {
    "annotation_path": "./data/tokenmask_data_256x2_cot_format/mask_understanding_sam_zoom_in_2157k_clean_repeat_pattern.json",
    "data_path": "",
}

MASK_UNDERSTANDING_GAR_RELATION = {
    "annotation_path": "./data/tokenmask_data_256x2_cot_format/mask_understanding_gar_relation510k.json",
    "data_path": "",
}

MASK_GENERATION_GRES_NO_TARGET = {
    "annotation_path": "./data/tokenmask_data_256x2/gres_no_target_cold_start_data14k.json",
    "data_path": "",
}

MASK_GENERATION_PADT_REFCOCO_DIRECTION_ORDER = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_padt_refcoco_direction_order_100k.json",
    "data_path": "",
}

MASK_GENERATION_REFSEG_DIRECTION_ORDER = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_refseg_direction_order_243k.json",
    "data_path": "",
}

MASK_GENERATION_PNG = {
    "annotation_path": "./data/tokenmask_data_256x2/mask_generation_png134k.json",
    "data_path": "",
}


ABLATION_SEGLLM_256X4 = {
    "annotation_path": "./data/ablation_data/segllm_256x4_1049k.json",
    "data_path": "",
}

ABLATION_SEGLLM_1024X4 = {
    "annotation_path": "./data/ablation_data/segllm_1024x4_959k.json",
    "data_path": "",
}

ABLATION_RES_256X4 = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_256x4_321k.json",
    "data_path": "",
}

ABLATION_RES_256X4_DIRECTION_ORDER = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_256x4_direction_order_100k.json",
    "data_path": "",
}

ABLATION_RES_1024X1 = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_1024x1_321k.json",
    "data_path": "",
}

ABLATION_RES_1024X1_DIRECTION_ORDER = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_1024x1_direction_order_100k.json",
    "data_path": "",
}

ABLATION_RES_65536X1 = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_65536x1_321k.json",
    "data_path": "",
}

ABLATION_RES_65536X1_DIRECTION_ORDER = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_65536x1_direction_order_100k.json",
    "data_path": "",
}

ABLATION_RES_1024X4 = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_1024x4_321k.json",
    "data_path": "",
}

ABLATION_RES_1024X4_DIRECTION_ORDER = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_1024x4_direction_order_100k.json",
    "data_path": "",
}

ABLATION_RES_1024X2 = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_1024x2_321k.json",
    "data_path": "",
}

ABLATION_RES_1024X2_DIRECTION_ORDER = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_1024x2_direction_order_100k.json",
    "data_path": "",
}

ABLATION_RES_512X2 = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_512x2_321k.json",
    "data_path": "",
}

ABLATION_RES_512X2_DIRECTION_ORDER = {
    "annotation_path": "./data/ablation_data/mask_generation_padt_refcoco_512x2_direction_order_100k.json",
    "data_path": "",
}

COLD_START_BRIEF_GCG = {
    "annotation_path": "./cold_start_data/gcg_cold_start_data_with_evidence_3k.json",
    "data_path": "",
}

COLD_START_DETAIL_GCG = {
    "annotation_path": "./cold_start_data/coconut_dw_cold_start_data9k.json",
    "data_path": "",
}

COLD_START_GRES = {
    "annotation_path": "./cold_start_data/gres_cold_start_data12k.json",
    "data_path": "",
}

COLD_START_NO_TARGET_GRES = {
    "annotation_path": "./cold_start_data/gres_no_target_cold_start_data14k.json",
    "data_path": "",
}

COLD_START_VER = {
    "annotation_path": "./cold_start_data/ver_cold_start_data75k.json",
    "data_path": "",
}

data_dict = {
    "mask_generation_refseg": MASK_GENERATION_REFSEG,
    "mask_generation_invig": MASK_GENERATION_INVIG,
    "mask_generation_gcg": MASK_GENERATION_GCG,
    "mask_generation_grandf": MASK_GENERATION_GRANDF,
    "mask_generation_reasonseg": MASK_GENERATION_REASONSEG,
    "mask_generation_v3det": MASK_GENERATION_V3DET,
    "mask_understanding_dam": MASK_UNDERSTANDING_DAM,
    "mask_generation_dw": MASK_GENERATION_DW,
    "mask_generation_coconut": MASK_GENERATION_COCONUT,
    "mask_generation_padt_refcoco": MASK_GENERATION_PADT_REFCOCO,
    "mask_generation_padt_ric": MASK_GENERATION_PADT_RIC,
    "mask_generation_padt_cocoseg": MASK_GENERATION_PADT_COCOSEG,
    "mask_generation_segllm": MASK_GENERATION_SEGLLM,
    "mask_generation_gres": MASK_GENERATION_GRES,
    "mask_generation_lisa": MASK_GENERATION_LISA,
    "mask_generation_psg_v1": MASK_GENERATION_PSG_V1,
    "mask_generation_psg_v2": MASK_GENERATION_PSG_V2,
    "mask_generation_psg_v3": MASK_GENERATION_PSG_V3,
    "mask_generation_psg_v4": MASK_GENERATION_PSG_V4,
    "mask_generation_dw_gcg": MASK_GENERATION_DW_GCG,
    "mask_generation_gres_no_target": MASK_GENERATION_GRES_NO_TARGET,
    "mask_generation_padt_refcoco_direction_order": MASK_GENERATION_PADT_REFCOCO_DIRECTION_ORDER,
    "mask_generation_refseg_direction_order": MASK_GENERATION_REFSEG_DIRECTION_ORDER,
    "mask_understanding_dam_zoom_in": MASK_UNDERSTANDING_DAM_ZOOM_IN,
    "mask_understanding_sam_zoom_in": MASK_UNDERSTANDING_SAM_ZOOM_IN,
    "mask_understanding_gar_zoom_in": MASK_UNDERSTANDING_GAR_ZOOM_IN,
    "mask_understanding_gar_relation": MASK_UNDERSTANDING_GAR_RELATION,
    "mask_generation_png": MASK_GENERATION_PNG,
    "ablation_256x4_segllm": ABLATION_SEGLLM_256X4,
    "ablation_1024x4_segllm": ABLATION_SEGLLM_1024X4,
    "ablation_256x4_res": ABLATION_RES_256X4,
    "ablation_256x4_res_direction_order": ABLATION_RES_256X4_DIRECTION_ORDER,
    "ablation_1024x1_res": ABLATION_RES_1024X1,
    "ablation_1024x1_res_direction_order": ABLATION_RES_1024X1_DIRECTION_ORDER,
    "ablation_65536x1_res": ABLATION_RES_65536X1,
    "ablation_65536x1_res_direction_order": ABLATION_RES_65536X1_DIRECTION_ORDER,
    "ablation_1024x4_res": ABLATION_RES_1024X4,
    "ablation_1024x4_res_direction_order": ABLATION_RES_1024X4_DIRECTION_ORDER,
    "ablation_1024x2_res": ABLATION_RES_1024X2,
    "ablation_1024x2_res_direction_order": ABLATION_RES_1024X2_DIRECTION_ORDER,
    "ablation_512x2_res": ABLATION_RES_512X2,
    "ablation_512x2_res_direction_order": ABLATION_RES_512X2_DIRECTION_ORDER,
    "cold_start_brief_gcg": COLD_START_BRIEF_GCG,
    "cold_start_detail_gcg": COLD_START_DETAIL_GCG,
    "cold_start_gres": COLD_START_GRES,
    "cold_start_no_target_gres": COLD_START_NO_TARGET_GRES,
    "cold_start_ver": COLD_START_VER,
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