import os
import json
import tqdm
import random
import tensorflow as tf
import re

def read_tfrecord_files(file_pattern, feature_description=None, num_parallel_reads=4, batch_size=32,
                        shuffle_buffer_size=10000):
    """
    读取TFRecord文件并以字典形式返回数据

    参数:
        file_pattern: TFRecord文件路径模式，支持通配符
        feature_description: 特征解析字典，默认为None(自动检测简单特征)
        num_parallel_reads: 并行读取文件数
        batch_size: 批次大小
        shuffle_buffer_size: 打乱数据的缓冲区大小，0表示不打乱

    返回:
        tf.data.Dataset对象，每个元素是包含特征的字典
    """
    # 获取文件列表
    file_paths = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle_buffer_size > 0)

    # 自动检测特征描述(如果未提供)
    if feature_description is None:
        feature_description = _detect_feature_description(file_paths.take(1))

    # 创建数据集
    dataset = file_paths.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_parallel_reads,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 打乱数据(如果需要)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)

    # 解析TFRecord
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 批处理
    dataset = dataset.batch(batch_size)

    # 预取数据
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def _detect_feature_description(file_paths):
    """自动检测TFRecord文件中的特征结构"""
    # 创建TFRecord读取器
    raw_dataset = tf.data.TFRecordDataset(file_paths)

    # 获取第一个样本
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

    # 构建特征描述
    feature_description = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof('kind')
        if kind == 'bytes_list':
            feature_description[key] = tf.io.FixedLenFeature([], tf.string)
        elif kind == 'float_list':
            feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
        elif kind == 'int64_list':
            feature_description[key] = tf.io.FixedLenFeature([], tf.int64)

    return feature_description

def split_list_random(lst, k=4000, seed=None):
    """
    从 lst 中随机选取 k 个元素作为子集A，剩余作为子集B。
    不放回抽样，A 与 B 不重叠。
    """
    if seed is not None:
        random.seed(seed)  # 可复现实验
    n = len(lst)
    if k > n:
        raise ValueError(f"k={k} 大于列表长度 n={n}")
    indices = random.sample(range(n), k)  # 随机挑 k 个索引
    indices_set = set(indices)
    subset_A = [lst[i] for i in indices]
    subset_B = [lst[i] for i in range(n) if i not in indices_set]
    return subset_A, subset_B

def main():
    all_data_dict = []
    for json_file in os.listdir('./temp_data_256x2_0927/ver'):
        if not json_file.endswith('.json'):
            continue
        with open(os.path.join('./temp_data_256x2_0927/ver', json_file), 'r') as f:
            json_data = json.load(f)
            all_data_dict.extend(json_data)
    
    rl_subset, cold_start_subset = split_list_random(all_data_dict, k=4000, seed=42)

    # rl subset
    rl_subset_image_keys = [item['image'] for item in rl_subset]
    rl_subset_dict = {item['image']: item for item in rl_subset}

    dataset = read_tfrecord_files("./data/HarborYuan/VisualReasoningTracer/ver_training_*.tfrecord", shuffle_buffer_size=0)
    total_num = 0
    for batch in dataset:
        total_num += 1
    for batch in tqdm.tqdm(dataset, total=total_num, desc="Processing"):
        ann_tensor = batch['json']
        annotations = [json.loads(s.decode("utf-8")) for s in ann_tensor.numpy()]
        for anno in annotations:
            image_name = anno['image_name']
            image_path = os.path.join('./data/sam_full', image_name)
            if image_path not in rl_subset_image_keys:
                continue

            original_annotation = anno['original_annotation']
            generated_ver_data = anno['generated_ver_data']

            objects_anns = original_annotation['objects_anns']

            candidates_ver_samples = generated_ver_data['candidates']

            assert len(candidates_ver_samples) == 1
            reasoning = candidates_ver_samples[0]['reasoning']
            reasoning_obj_ids = re.findall(r"<ver>(<obj\d+>)</ver>", reasoning)
            answer_obj_ids = candidates_ver_samples[0]['answer_objects']

            reasoning_masks = [objects_anns[obj_id]['segmentation'] for obj_id in reasoning_obj_ids]
            answer_masks = [objects_anns[obj_id]['segmentation'] for obj_id in answer_obj_ids]
            rl_subset_dict[image_path].update({'reasoning_masks': reasoning_masks, 'answer_masks': answer_masks})
    rl_subset = list(rl_subset_dict.values())
    with open(f'./cold_start_data/ver_rl_source{len(rl_subset)}k.json', 'w') as f:
        json.dump(rl_subset, f, indent=4)
    
    with open(f"./cold_start_data/ver_cold_start_data{len(cold_start_subset)//1000}k.json", 'w') as f:
        json.dump(cold_start_subset, f, indent=4)

if __name__ == "__main__":
    main()
    

