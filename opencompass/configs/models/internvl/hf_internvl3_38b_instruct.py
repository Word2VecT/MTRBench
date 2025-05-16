from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internvl3-38b-instruct-turbomind',
        path='OpenGVLab/InternVL3-38B-Instruct',
        engine_config=dict(session_len=4096, max_batch_size=1, tp=2),
        gen_config=dict(temperature=0, max_new_tokens=4096),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=2),
    )
]