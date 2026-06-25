import os
import re
import copy
import json
import random
import numpy as np
from PIL import Image
from typing import Dict, Optional, Sequence, List, Tuple, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
# from projects.vlm.tokenmask.models import PerceptionLMProcessor

from projects.vlm.tokenmask.datasets.data import data_list


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def _normalize_conversations(obj):
    # 如果是字符串，尽量先解析为 Python 对象
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            # 保底兜底：包成 assistant 一条
            return [{"from": "human", "value": obj}]

    # 如果是单条 dict，包成 list
    if isinstance(obj, dict):
        return [obj]

    # 如果已经是 list，确认每一项都是 dict；不是的话兜底包装
    if isinstance(obj, list):
        fixed = []
        for x in obj:
            if isinstance(x, dict):
                fixed.append(x)
            elif isinstance(x, str):
                # 尝试解析字符串消息
                try:
                    xj = json.loads(x)
                    fixed.append(xj if isinstance(xj, dict) else {"from": "human", "value": x})
                except Exception:
                    fixed.append({"from": "human", "value": x})
            else:
                # 其他类型兜底
                fixed.append({"from": "human", "value": str(x)})
        return fixed

    # 其他类型兜底
    return [{"from": "human", "value": str(obj)}]

def convert_interleaved_prompt_to_message(
    prompt: str,
    image_data: List[str],
    image_placeholder: str = "<image>"
) -> List[Dict[str, Any]]:
    """
    将交错的图文prompt字符串转换为多模态模型的标准message格式。
    Args:
        prompt (str): 包含文本和图像占位符的字符串。
                      例如: "这是第一张图<image>\n这是第二张图<image>，请比较它们。"
        image_data (List[str]): 一个包含图像数据（Base64编码或URL）的列表。
                                列表的顺序应与prompt中占位符的出现顺序一致。
        image_placeholder (str, optional): prompt中使用的图像占位符。默认为 "<image>"。
    Returns:
        List[Dict[str, Any]]: 转换后的标准message格式列表。
                               例如: [{"role": "user", "content": [...]}]
    Raises:
        ValueError: 如果prompt中的图像占位符数量与image_data列表的长度不匹配。
    """
    # 1. 检查占位符数量和图像数据数量是否匹配
    num_placeholders = prompt.count(image_placeholder)
    if num_placeholders != len(image_data):
        print("=======>>>", prompt)    
        raise ValueError(
            f"Prompt中的图像占位符数量 ({num_placeholders}) 与提供的图像数据数量 ({len(image_data)}) 不匹配。"
        )
    if len(image_data) == 0:
        message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
        return message

    # 2. 使用正则表达式分割prompt字符串
    # re.split会保留分割符，我们需要一个能同时捕获文本和占位符的模式
    # (image_placeholder) -> 捕获组，使分割符保留在结果中
    parts = re.split(f'({re.escape(image_placeholder)})', prompt)
    content_list = []
    image_index = 0
    # 3. 遍历分割后的部分，构建content列表
    for part in parts:
        if not part:  # 跳过因分割产生的空字符串
            continue
        if part == image_placeholder:
            # 这是一个图像部分
            content_list.append({
                "type": "image",
                "image": image_data[image_index]
            })
            image_index += 1
        else:
            # 这是一个文本部分
            text_content = part
            # 检查前一个元素是否是图像，如果是，则处理前导的 '\n'
            if content_list and content_list[-1]["type"] == "image":
                if text_content.startswith('\n'):
                    text_content = text_content[1:] # 去掉开头的换行符
            # 如果处理后文本不为空，则添加
            if text_content:
                content_list.append({
                    "type": "text",
                    "text": text_content
                })
    # 4. 封装成最终的message格式
    message = {
        "role": "user",
        "content": content_list
    }
    return message


class TokenMaskDataset(Dataset):
    def __init__(
        self,
        dataset_use: str = "",
        model_path: str = "",
        max_num_tiles=12,
    ):
        super(TokenMaskDataset, self).__init__()

        dataset = dataset_use.split(",")
        dataset_list = data_list(dataset)

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                with open(data["annotation_path"], "r") as f:
                    annotations = json.load(f)
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
                annotations = annotations * int(sampling_rate)
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        self.list_data_dict = list_data_dict
        self.max_num_tiles = max_num_tiles
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        # self.processor = PerceptionLMProcessor.from_pretrained(model_path, use_fast=False)
        self.processor.image_processor.max_num_tiles = self.max_num_tiles

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            new_conversations = _normalize_conversations(sample["conversations"])
            length_list.append(
                sum(len(conv["value"].split()) for conv in new_conversations)
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            new_conversations = _normalize_conversations(sample["conversations"])
            cur_len = sum(
                len(conv["value"].split()) for conv in new_conversations
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def modality_length(self):
        return self.modality_lengths

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))
        
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def parse_label(self,labels):

        start_tokens = torch.tensor([128006, 78191, 128007, 271], device=labels.device)  # 起始
        end_token = 128009  # 终止

        labels = labels.clone()
        mask = torch.full_like(labels, fill_value=-100)

        i = 0
        while i < len(labels):
            # 找到 start pattern
            if i + len(start_tokens) <= len(labels) and torch.equal(labels[i:i+len(start_tokens)], start_tokens):
                start = i + len(start_tokens)  # 跳过 start specials
                try:
                    end = (labels[start:] == end_token).nonzero(as_tuple=True)[0][0].item() + start
                except IndexError:
                    break  # 没找到终止符
                # 保留 [start:end] + end 本身
                if end >= start:
                    mask[start:end+1] = labels[start:end+1]
                i = end + 1
            else:
                i += 1

        return mask
    
    def get_data(self, input_sources):
        assert len(input_sources) == 1

        new_sources = []
        for source in input_sources:
            copied_source = {}
            for k, v in source.items():
                if k == "image" and len(v) > 0 and v is not None:
                    copied_source[k] = copy.deepcopy(v)
                elif k == "image":
                    continue
                else:
                    copied_source[k] = copy.deepcopy(v)
            new_sources.append(copied_source)
        sources = new_sources

        if "image" in sources[0] and len(sources[0]["image"]) != 0:
            image_folder = sources[0]["data_path"]
            image_file = sources[0]["image"]

            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                else:
                    image_file = image_file[0]
                    image_file = [os.path.join(image_folder, image_file)]
                    
            else:
                image_file = [os.path.join(image_folder, image_file)]
            images = [Image.open(image_path).convert('RGB') for image_path in image_file]
        if "video" in sources[0]:
            return None
        if "image" not in sources[0] and "video" not in sources[0]:
            images = []

        chat_sources = []
        for e in sources:
            convs = _normalize_conversations(e["conversations"])
            chat_sources.append(convs)
        assert len(chat_sources) == 1
        source = chat_sources[0]
        if isinstance(source, str):
            source = json.loads(source)
        try:
            if source[0]["from"] != "human":
                source = source[1:]
        except:
            print(sources)
        
        messages= []
        for conv_i, conv in enumerate(source):
            role = conv['from']
            content = conv['value']
            if role == 'human' and conv_i < 2:
                msg = convert_interleaved_prompt_to_message(content, images, image_placeholder='<image>')
            elif role == 'human':
                msg = convert_interleaved_prompt_to_message(content, [], image_placeholder='<image>')
            else:
                msg = {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": content}]
                }
            messages.append(msg)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            # return_tensors="pt",
            return_dict=True,
            disable_grouping=True,
        )

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        labels = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        # print("==========>pixel_values: ", len(inputs["pixel_values"]), [v.shape for v in inputs["pixel_values"]])
        # exit(0)

        # print("========>input_ids: ", input_ids.shape)
        # print("========>attention_mask: ", attention_mask.shape)
        # exit(0)

        # input_ids torch.Size([1, 1956])
        # attention_mask torch.Size([1, 1956])
        # pixel_values torch.Size([1, 7, 3, 448, 448])
        # print("===========>>>", messages)
        # print("========>input_ids: ", len(inputs['input_ids']), [len(v) for v in inputs['input_ids']])
        # print("========>attention_mask: ", len(inputs['attention_mask']), [len(v) for v in inputs['attention_mask']])
        # print("========>pixel_values: ", len(inputs["pixel_values"]), [v.shape for v in inputs["pixel_values"]])
        # exit(0)

        pixel_values = inputs["pixel_values"]
        # input_ids = inputs["input_ids"].squeeze(0)
        # labels = inputs["input_ids"].squeeze(0).clone()
        # attention_mask = inputs["attention_mask"].squeeze(0)
        input_ids = input_ids.squeeze(0)
        labels = labels.squeeze(0).clone()
        labels = self.parse_label(labels)
        attention_mask = attention_mask.squeeze(0)

        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
      
        return ret

    
    def _get_item(self, i):

        sources = self.list_data_dict[i]
        assert isinstance(sources, dict)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        return self.get_data(sources)
        
    def __getitem__(self, i):
        num_base_retries = 3
        num_final_retries = 30

        _max_refetch = 1000
        index = i
        for _ in range(_max_refetch + 1):
            # try:
            #     sample = self._get_item(index)
            #     if sample is None:
            #         print(
            #             f"[Try other #{_+1}] Failed to fetch sample {i}."
            #         )
            #         index = self._rand_another()
            #     else:
            #         return sample
            # except Exception as e:
            #     print(
            #         f"[Try other #{_+1}] Failed to fetch sample {i}. Exception:",
            #         e,
            #     )
            #     index = self._rand_another()
            sample = self._get_item(index)
            if sample is None:
                print(
                    f"[Try other #{_+1}] Failed to fetch sample {i}."
                )
                index = self._rand_another()
            else:
                return sample