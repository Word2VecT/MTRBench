from torch import mode
from opencompass.models import OpenAI


# model_name = 'Qwen/Qwen2.5-VL-72B-Instruct'
# model_name = 'Qwen/Qwen2.5-VL-32B-Instruct'
# model_name = 'Pro/Qwen/Qwen2.5-VL-7B-Instruct'
# model_name = 'doubao-1-5-pro-32k-250115'
# model_name = 'doubao-1-5-vision-pro-250328'

# model_name = 'doubao-1-5-thinking-pro-250415'
# model_name = 'doubao-1-5-thinking-pro-m-250415'

# model_name = 'Qwen/QwQ-32B'
# model_name = 'Qwen/QVQ-72B-Preview'

# model_name = 'Qwen/Qwen2.5-7B-Instruct'
model_name = 'Qwen/Qwen2.5-14B-Instruct'
# model_name = 'Qwen/Qwen2.5-32B-Instruct'
# model_name = 'Qwen/Qwen2.5-72B-Instruct'

api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr=model_name,
        type=OpenAI, path=model_name,
        key='sk-IspF33lON8dbECXOTeDbyiPEKEjTmiKmsK3lP2I6ZEbr9gNg', # vip
        meta_template=api_meta_template,
        query_per_second=5,
        max_out_len=32768,
        # max_out_len=8192,
        # max_out_len=4096,
        batch_size=8,
        temperature=0.0),
]
