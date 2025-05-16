from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import FunnyDatasetV2, MATHPassKEvaluator, math_postprocess_v2


funnybench_reader_cfg = dict(input_columns=['task', 'clues'], output_column='answer')

# funnybench_all_sets = ['QA', 'Choices']
funnybench_all_sets = ['QA']

funnybench_datasets = []
for _name in funnybench_all_sets:
    funnybench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt="You are playing an escape room puzzle game, and you need to use clues to solve the puzzle in front of you. You must provide a single, definitive answer.\n\nPuzzle:\n{task}\n\nClues:\n{clues}\nLet's think step by step and put the final answer in \\boxed{{}}. Like this: \\boxed{{THE ANSWER}}."),
                ],
            )),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=32768, temperature=0.6, top_p=0.95))

    funnybench_eval_cfg = dict(
        evaluator=dict(type=MATHPassKEvaluator, k=5),
        pred_postprocessor=dict(type=math_postprocess_v2))

    funnybench_datasets.append(
        dict(
            abbr=f'funny_en_long_pass5_temp0.6_{_name}',
            num_repeats=5,
            type=FunnyDatasetV2,
            path=f'/mnt/petrelfs/tangzinan/COLM2025/dataset/LLM/EN/EN_{_name}.json',
            reader_cfg=funnybench_reader_cfg,
            infer_cfg=funnybench_infer_cfg,
            eval_cfg=funnybench_eval_cfg)
    )
    
del _name