import tensorflow as tf
import time
import json

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

def read_tfrecord(tfrecord_path):
    dataset = read_tfrecord_files(
        file_pattern=tfrecord_path,
        batch_size=1,
        shuffle_buffer_size=-1,
    )

    count = 0
    for batch in dataset:
        count += 1
    return dataset, count

def decode_tf_batch(batch):
    ret = {}
    for feature_name, feature_value in batch.items():
        if feature_name == "image":
            image = tf.io.decode_image(feature_value[0]).numpy()
            ret["image"] = image
        elif feature_name == "annotation":
            ann_dict = json.loads(feature_value[0].numpy().decode('utf-8'))
            ret["annotation"] = ann_dict
        elif feature_name == "key":
            image_name = feature_value[0].numpy().decode('utf-8')
            ret["image_name"] = image_name
    assert "image" in ret and "annotation" in ret
    return ret

if __name__ == '__main__':
    dataset = read_tfrecord_files("./data/tyfeld/coconut_dw/coconut_s_gcg_objcap_*.parquet")
    for sample in dataset:
        # dict_keys(['image', 'json', 'key'])
        # image = tf.io.decode_image(sample['image'][0]).numpy()
        # print(type(image))
        # print(image.shape)
        # exit(0)
        ann_tensor = sample['json']
        annotations = [json.loads(s.decode("utf-8")) for s in ann_tensor.numpy()]
        # for anno in annotations:
        #     print(anno.keys()) # dict_keys(['image_name', 'annotation_file', 'original_annotation', 'generated_ver_data', 'timestamp', 'raw_response', 'meta_data'])
        #     exit(0)
        print(annotations[0])
        exit(0)
        # print("image: ", type(sample['image']))
        # print("json: ", sample['json'])
        # print("key: ", sample['key'])
        # exit(0)