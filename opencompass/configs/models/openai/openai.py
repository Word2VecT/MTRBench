from torch import mode
from opencompass.models import OpenAI

# model_name = 'gpt-4o-mini'
# model_name = 'gpt-4.1-mini-2025-04-14'
# model_name = 'gpt-4.1-2025-04-14'
# model_name = 'chatgpt-4o-latest'
# model_name = 'claude-3-7-sonnet-latest'
# model_name = 'grok-3-beta'
# model_name = 'USD-guiji/deepseek-v3'
# model_name = 'llama-3.3-70b-instruct'
# model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
# model_name = 'llama-3.2-11b-vision-instruct'
# model_name = 'meta-llama/llama-3.2-90b-vision-instruct'

model_name = 'llama-4-maverick'
# model_name = 'llama-4-scout'

api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr=model_name,
        type=OpenAI, path=model_name,
        key='sk-1OPT9ZVr087cDAbAkEngsBjeMAcbbCNSlrUVJgijTAPOoQdp',  # New
        # key='xai-NGIQp7EGScw3F6cl5OlTyjCMbWLVTR2iWRlfqRqPkN0i3dtsMyPn56otMhX245bujcbhi3aSKbdaZ262',  # Grok
        meta_template=api_meta_template,
        query_per_second=10,
        max_out_len=4096,
        batch_size=8,
        temperature=0.6),
]
