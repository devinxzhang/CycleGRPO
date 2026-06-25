from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR, OptimWrapper
from torch.optim import AdamW

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop

from transformers import AutoProcessor, AutoTokenizer

from projects.transformers.vq_sam2 import VQ_SAM2Config, SAM2Config
from projects.vlm.qwen2_5_vl_vq_sam2.models.qwen2_5_vl_vq_sam2 import Qwen2_5_VL_VQ_SAM2
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.llava_vqa_dataset import LLaVADataset
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.collect_fns import qwen2_5_vl_vq_sam2_collate_fn, qwen2_5_vl_vq_sam2_collate_fn_data_flatten

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
path = './pretrained_weights/qwen2_5_vl_vq_sam2_3b'

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 8
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 2
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)


#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# sam2_config = dict(
#     type=SAM2Config,
#     cfg_path="sam2.1_hiera_l.yaml",
#     ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
# )

vq_sam2_config = dict(
    codebook_size=1024,
    codebook_depth=4,
    shared_codebook=False,
    codebook_latent_dim=256,
)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

qwen_processor = dict(
    type=AutoProcessor.from_pretrained,
    pretrained_model_name_or_path=path,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28
)


data_flatten = False
model = dict(
    type=Qwen2_5_VL_VQ_SAM2,
    vq_sam2_config=vq_sam2_config,
    tokenizer=tokenizer,
    preprocessor=qwen_processor,
    base_mllm_path=None,
    mllm_path=path,
    freeze_llm=False,
    freeze_visual_encoder=True,
    data_flatten=data_flatten,
)


#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

llava_665k_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='./data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    image_folder='./data/llava_data/llava_images/',
    max_length=8192,
    preprocessor=qwen_processor,
    repeats=1,
)

mask_generation_insseg335k_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='./data/vq_sam2_data/mask_generation_insseg335k.json',
    image_folder=None,
    max_length=8192,
    preprocessor=qwen_processor,
    repeats=1,
)

mask_understanding_dam963k_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='./data/vq_sam2_data/mask_understanding_dam963k.json',
    image_folder=None,
    max_length=8192,
    preprocessor=qwen_processor,
    repeats=1,
)

mask_generation_gcg39k_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='./data/vq_sam2_data/mask_generation_gcg39k.json',
    image_folder=None,
    max_length=8192,
    preprocessor=qwen_processor,
    repeats=1,
)

mask_generation_reasonseg20k_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='./data/vq_sam2_data/mask_generation_reasonseg2k.json',
    image_folder=None,
    max_length=8192,
    preprocessor=qwen_processor,
    repeats=1,
)

mask_generation_refseg67k_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='./data/vq_sam2_data/mask_generation_refseg67k.json',
    image_folder=None,
    max_length=8192,
    preprocessor=qwen_processor,
    repeats=1,
)


train_dataset = dict(
    type=ConcatDataset, datasets=[
        # llava_665k_dataset, 
        mask_generation_insseg335k_dataset, 
        mask_understanding_dam963k_dataset,
        # mask_generation_refseg67k_dataset,
    ]
)

collate_fn = qwen2_5_vl_vq_sam2_collate_fn_data_flatten if data_flatten else qwen2_5_vl_vq_sam2_collate_fn
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    # loss_scale='dynamic',
    # dtype='bfloat16',
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
# visualizer = None
from mmengine.visualization import Visualizer, TensorboardVisBackend
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
