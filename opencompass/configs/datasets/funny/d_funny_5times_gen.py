from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import FunnyDatasetV2, MATHEvaluator, math_postprocess_v2


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
                    dict(role='HUMAN', prompt="你正在进行一场密室逃脱游戏，你需要利用线索，解开眼前的谜题，你必须给出唯一的确定答案。\n\n谜题：\n{task}\n\n线索：\n{clues}\n{choices}\n让我们一步一步思考，并将最终答案放在 \\boxed{{}} 中。"),
                ],
            )),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=4096, temperature=0.0))

    funnybench_eval_cfg = dict(
        evaluator=dict(type=MATHEvaluator),
        pred_postprocessor=dict(type=math_postprocess_v2))

    funnybench_datasets.append(
        dict(
            abbr=f'funny_{_name}_5times',
            num_repeats=5,
            type=FunnyDatasetV2,
            path=f'/mnt/petrelfs/tangzinan/COLM2025/dataset/LLM/CH/CH_{_name}.json',
            reader_cfg=funnybench_reader_cfg,
            infer_cfg=funnybench_infer_cfg,
            eval_cfg=funnybench_eval_cfg)
    )
    
del _name