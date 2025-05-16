from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='QVQ-72B-Preview-hf',
        path='Qwen/QVQ-72B-Preview',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
