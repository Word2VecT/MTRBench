from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_1-nemotron-nano-8b-v1-hf',
        path='nvidia/Llama-3.1-Nemotron-Nano-8B-v1',
        max_out_len=32768,
        batch_size=1,
        run_cfg=dict(num_gpus=1)
    )
]