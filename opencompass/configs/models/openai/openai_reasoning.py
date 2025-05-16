from torch import mode
from opencompass.models import OpenAI

# model_name = 'claude-3-7-sonnet-thinking'
# model_name = 'google/gemini-2.5-pro-preview'
model_name = 'gemini-2.5-pro-preview-05-06'
# model_name = 'google/gemini-2.5-flash-preview'
# model_name = 'gemini-2.5-pro-preview-05-06'
# model_name = 'o4-mini'
# model_name = 'grok-3-mini-beta'
# model_name = 'USD-guiji/deepseek-r1'

api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr=model_name,
        type=OpenAI, path=model_name,
        # key='sk-GyHHUeCpfDMWqBsegY2JTuyregEzQjDrRFZdbVW862LYVJt5',  # Old
        key='sk-1OPT9ZVr087cDAbAkEngsBjeMAcbbCNSlrUVJgijTAPOoQdp',  # New
        # key='xai-NGIQp7EGScw3F6cl5OlTyjCMbWLVTR2iWRlfqRqPkN0i3dtsMyPn56otMhX245bujcbhi3aSKbdaZ262',  # Grok
        meta_template=api_meta_template,
        query_per_second=10,
        max_out_len=32768,
        # max_out_len=8192,
        batch_size=8,
        temperature=0.6),
]
