# from opencompass.models import HuggingFacewithChatTemplate

# models = [
#     dict(
#         type=HuggingFacewithChatTemplate,
#         abbr='InternVL3-8B-Instruct',
#         path='OpenGVLab/InternVL3-8B-Instruct',
#         max_out_len=4096,
#         max_seq_len=4096,
#         batch_size=8,
#         run_cfg=dict(num_gpus=1),
#     )
# ]

from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internvl3-8b-instruct-turbomind',
        path='OpenGVLab/InternVL3-8B-Instruct',
        engine_config=dict(session_len=4096, max_batch_size=16, tp=1),
        gen_config=dict(temperature=0, max_new_tokens=4096),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]