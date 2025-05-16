from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepcoder-14b-preview',
        path='agentica-org/DeepCoder-14B-Preview',
        max_out_len=32768,
        batch_size=1,
        run_cfg=dict(num_gpus=1)
    )
]