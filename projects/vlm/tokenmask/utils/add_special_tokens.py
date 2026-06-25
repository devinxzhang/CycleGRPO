

if __name__ == "__main__":
    # from transformers import AutoProcessor, AutoModelForImageTextToText
    # from tokenizers import AddedToken
    # import torch, os

    # model_id = "facebook/Perception-LM-8B"
    # save_dir = "facebook/Perception-LM-8B-MT256x2"

    # # 1) 扩 tokenizer
    # MT_START_TOKEN = '<|mt_start|>'
    # MT_END_TOKEN = '<|mt_end|>'
    # MT_CONTEXT_TOKEN = '<|mt_{}|>'
    # new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256+256)] + [MT_END_TOKEN]

    # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # tokenizer = processor.tokenizer
    # added = tokenizer.add_tokens(
    #     [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
    #     special_tokens=False,
    # )
    # print("added:", added)
    # os.makedirs(save_dir, exist_ok=True)
    # processor.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

    # # 2) 扩模型词表并初始化新增行
    # model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    # model.resize_token_embeddings(len(tokenizer))

    # with torch.no_grad():
    #     emb = model.get_input_embeddings().weight
    #     old_vocab = emb.shape[0] - added
    #     mu = emb[:old_vocab].mean(0, keepdim=True)
    #     std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
    #     emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    # model.save_pretrained(save_dir)
    # print("Saved to", save_dir)


    # from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    # from tokenizers import AddedToken
    # import torch, os

    # model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # save_dir = "Qwen/Qwen3-VL-30B-A3B-MT-256x2"

    # # 1) 扩 tokenizer
    # MT_START_TOKEN = '<|mt_start|>'
    # MT_END_TOKEN = '<|mt_end|>'
    # MT_CONTEXT_TOKEN = '<|mt_{}|>'
    # new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256*2)] + [MT_END_TOKEN]
    # # new_tokens = ['[SEG]']

    # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # tokenizer = processor.tokenizer
    # added = tokenizer.add_tokens(
    #     [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
    #     special_tokens=False,
    # )
    # print("added:", added)
    # os.makedirs(save_dir, exist_ok=True)
    # processor.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

    # # 2) 扩模型词表并初始化新增行
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    # model.resize_token_embeddings(len(tokenizer))

    # with torch.no_grad():
    #     emb = model.get_input_embeddings().weight
    #     old_vocab = emb.shape[0] - added
    #     mu = emb[:old_vocab].mean(0, keepdim=True)
    #     std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
    #     emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    # model.save_pretrained(save_dir)
    # print("Saved to", save_dir)


    # from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
    from tokenizers import AddedToken
    import torch, os
    import math

    TP = 4
    MAKE_DIVISIBLE_BY = 128

    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    save_dir = "Qwen/Qwen3-VL-30B-A3B-MT-256x2"

    # 1) 扩 tokenizer
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256*2)] + [MT_END_TOKEN]
    # new_tokens = ['[SEG]']

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    added = tokenizer.add_tokens(
        [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
        special_tokens=False,
    )
    print("added:", added)

    # 1.1) 计算当前 vocab 大小，并向上补齐到能同时被 TP 和 MAKE_DIVISIBLE_BY 整除的值
    orig_vocab_size = len(tokenizer)
    # 先对齐到 MAKE_DIVISIBLE_BY 的倍数
    target_vocab_size = math.ceil(orig_vocab_size / MAKE_DIVISIBLE_BY) * MAKE_DIVISIBLE_BY
    # 再确保能被 TP 整除（由于 128 本身是 4 的倍数，这一步一般已经满足；否则额外再向上修正）
    if target_vocab_size % TP != 0:
        target_vocab_size = math.ceil(target_vocab_size / TP) * TP
    pad_needed = target_vocab_size - orig_vocab_size
    if pad_needed > 0:
        pad_tokens = [f"<|pad_extra_{i}|>" for i in range(pad_needed)]
        tokenizer.add_tokens(
            [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in pad_tokens],
            special_tokens=False,
        )
        print(f"Padded {pad_needed} tokens to reach {len(tokenizer)} (TP-divisible)")


    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 2) 扩模型词表并初始化新增行
    # model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    # model.resize_token_embeddings(len(tokenizer))
    final_vocab_size = len(tokenizer)  # 一定要用对齐后的最终大小
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(final_vocab_size)

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old_vocab = emb.shape[0] - added
        mu = emb[:old_vocab].mean(0, keepdim=True)
        std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
        emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    model.save_pretrained(save_dir)
    print("Saved to", save_dir)


