from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='QwQ-32B-hf',
        path='Qwen/QwQ-32B',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
