datasets = [
    [
        dict(
            abbr='funny_QA_pass5_temp0.6',
            eval_cfg=dict(
                evaluator=dict(
                    k=5, type='opencompass.datasets.MATHPassKEvaluator'),
                pred_postprocessor=dict(
                    type='opencompass.datasets.math_postprocess_v2')),
            infer_cfg=dict(
                inferencer=dict(
                    max_out_len=4096,
                    temperature=0.6,
                    top_p=0.95,
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            '你正在进行一场密室逃脱游戏，你需要利用线索，解开眼前的谜题，你必须给出唯一的确定答案。\n\n谜题：\n{task}\n\n线索：\n{clues}\n{choices}\n让我们一步一步思考，并将最终答案放在 \\boxed{{}} 中。',
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            num_repeats=5,
            path='/mnt/petrelfs/tangzinan/COLM2025/dataset/LLM/CH/CH_QA.json',
            reader_cfg=dict(
                input_columns=[
                    'task',
                    'clues',
                ], output_column='answer'),
            type='opencompass.datasets.FunnyDatasetV2'),
    ],
]
eval = dict(runner=dict(task=dict()))
models = [
    dict(
        abbr='InternVL3-8B',
        batch_size=8,
        max_out_len=4096,
        path='OpenGVLab/InternVL3-8B',
        run_cfg=dict(num_gpus=1),
        type='opencompass.models.HuggingFacewithChatTemplate'),
]
work_dir = 'outputs/default/20250515_140725'
