import argparse
import math
import os
import torch
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import copy

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils import _init_dist_pytorch, get_dist_info, collect_results_cpu
from PIL import Image
import re
import json

from xtuner.model.utils import guess_load_checkpoint

from qwen_vl_utils import process_vision_info
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize

def parse_args():
    parser = argparse.ArgumentParser(description='GCG')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/vq_sam2_2M/iter_44916.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--save_dir',
        default='./gcg_pred/',
        help='save path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

IMAGE_FOLDER = './data/glamm_data/images/grandf/val_test/'


class GCGInferenceDataset:
    def __init__(self,
                 image_folder,
                 save_dir=None,
                 ):
        self.image_folder = image_folder

        self.images = os.listdir(image_folder)

        if save_dir is not None:
            # filter evaluated
            self.save_dir = save_dir
            exsits_files = os.listdir(self.save_dir)
            exsits_files = [_file[:-5] for _file in exsits_files]
            _images = []
            for i, item in enumerate(self.images):
                if item[:-4] not in exsits_files:
                    _images.append(item)
            self.images = _images

    def __len__(self):
        return len(self.images)

    def get_questions(self):
        question = "Could you please give me a brief description of the image? Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."
        return question

    def __getitem__(self, index):
        data_dict = {}
        questions = self.get_questions()
        image_file = self.images[index]
        data_dict['image_file'] = image_file

        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        data_dict['image'] = image
        data_dict['text'] = "<image>\n" + questions

        data_dict['img_id'] = image_file
        return data_dict

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]


def remove_special_tokens(text):
    pattern = r"<\|mt_(start|end|\d{4})\|>"
    return re.sub(pattern, "", text)


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1


    # build qwen25vl model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )
    
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_state_dict = guess_load_checkpoint(args.vq_sam2_path)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    dataset = GCGInferenceDataset(
        image_folder=IMAGE_FOLDER,
        save_dir=args.save_dir,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        w, h = data_batch['image'].size

        image_file= prediction['image_file']
        image_file = os.path.join(IMAGE_FOLDER, image_file)
        question = data_batch['text'].replace('<image>\n', '').strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # print(messages)
        # exit(0)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print("Assistant: ", output_text)

        quant_ids = extract_mt_token_ids(output_text[0])
        if len(quant_ids) == 0:
            print("No SEG !!!")
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
        else:
            assert len(quant_ids) % CODEBOOK_DEPTH == 0
            batch_size = len(quant_ids) // CODEBOOK_DEPTH
            remap_quant_ids = []
            for bs_id in range(batch_size):
                chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
                remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
                remap_quant_ids.append(remap_chunk_quant_ids)

            image = Image.open(image_file).convert('RGB')
            ori_width, ori_height = image.size
            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

            quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
            _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
            _pred_masks = _pred_masks > 0.5
            _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
            prediction['prediction_masks'] = torch.from_numpy(_pred_masks).to(torch.bool)

        process_and_save_output(
            args.save_dir,
            prediction['image_file'],
            output_text[0],
            prediction['prediction_masks']
        )
        results.append(output_text[0])

    results = collect_results_cpu(results, len(dataset), tmpdir='./gcg_eval_tmp')


def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<|object_ref_start|>(.*?)<|object_ref_end|>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    # cleaned_str = cleaned_str.replace('[SEG]', '')
    cleaned_str = remove_special_tokens(cleaned_str)

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    # print(f"clean_str: {cleaned_str}")
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    # print(cleaned_str)
    # print(phrases)

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle

if __name__ == '__main__':
    main()