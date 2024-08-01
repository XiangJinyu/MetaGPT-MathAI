# -*- coding: utf-8 -*-
# @Date    : 7/7/2024 17:07 PM
# @Author  : didi
# @Desc    : test on human eval graph
import datetime
import os
import json
import subprocess
import sys
import asyncio
import aiofiles
from metagpt.llm import LLM
from examples.ags.w_action_node.math_graph import Gsm8kGraph
from examples.ags.w_action_node.operator import GenerateCode, GenerateCodeBlock
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
import pandas as pd



generate_code = GenerateCode(llm=LLM())
generate_code_block = GenerateCodeBlock(llm=LLM())
solver = Gsm8kGraph(name="solver", llm=LLM())

from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM


async def sample_generate(id, result_path: str = "samples.jsonl", mode: str = "ags"):
    case = get_human_eval_plus()[f"{id}"]
    if mode == "ags":
        solution_result = await solver(case['prompt'], ensemble_count=5)
        sample_dict = dict(task_id=case['task_id'], solution=solution_result['final_solution'])
    elif mode == "alpha":
        solution_result = await solver.alpha_codium(case['task_id'], case['prompt'], ensemble_count=5)
        sample_dict = dict(task_id=case['task_id'], solution=solution_result['final_solution'])
    elif mode == "llm":
        solution_result = await generate_code_block(case['prompt'], case['entry_point'])
        sample_dict = dict(task_id=case['task_id'], solution=solution_result['code_solution'])
        print(sample_dict)
    with open(result_path, mode='a') as f:
        f.write(json.dumps(sample_dict) + '\n')
    jsonl_ranker(result_path, result_path)


def automatic_sanitize(result_path: str = "samples.jsonl"):
    """
    在命令行中自动执行 evalplus.sanitize --samples result_path
    返回result_path前缀加上"-sanitized.jsonl"
    """
    command = ["evalplus.sanitize", "--samples", result_path]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"执行命令时出错: {e}")
        return None

    # 构建sanitized文件路径
    base_name = os.path.splitext(result_path)[0]
    sanitized_path = f"{base_name}-sanitized.jsonl"

    return sanitized_path


def automatic_evalplus(result_path: str = "samples.jsonl"):
    """
    在命令行中自动执行 evalplus.evaluate --dataset humaneval --samples samples.jsonl --parallel 2 --base-only
    """
    command = [
        sys.executable,  # 使用当前 Python 解释器
        "-m",
        "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", result_path,
        "--parallel", "2",
        "--base-only"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("错误输出:", e.stderr)
        return False


def extract_failure_tests(file_path: str = "samples_eval_results.json"):
    with open(file_path, 'r') as f:
        task_results = json.load(f)

    failed_tests = []

    for task in task_results['eval'].values():
        if task[0]["base_status"] == "fail":
            failed_test = {
                "task_id": task[0]["task_id"],
                # "solution": task["solution"],
                # "fail_tests": task["base_fail_tests"]
            }
            failed_tests.append(failed_test)
    print(len(failed_tests))

    return failed_tests


# asyncio.run(sample_generate('HumanEval/101'))
# asyncio.run(samples_generate(mode='ags'))
# jsonl_ranker("samples.jsonl", "samples.jsonl")
# {"task_id": "HumanEval/101", "solution": "def words_string(s):\n    import re\n    return re.split(r'[,\\s]\\s*', s)"}

# if __name__ == '__main__':
#     class OpenAI(DeepEvalBaseLLM):
#         def __init__(self):
#             self.solver = Gsm8kGraph(name="solver", llm=LLM())
#
#         def load_model(self):
#             # 这里应该是加载模型的逻辑
#             pass
#
#         async def a_generate(self, prompt: str) -> str:
#             solution_result = await self.solver(prompt)
#             return solution_result['solution']
#
#         def generate(self, prompt: str) -> str:
#             loop = asyncio.get_event_loop()
#             solution_result = loop.run_until_complete(self.a_generate(prompt))  # 等待 a_generate 方法完成
#             return solution_result
#
#         def get_model_name(self):
#             return "Custom Azure OpenAI Model"
#
#
#     # Replace these with real values
#     openai = OpenAI()
#     # print(openai.generate("Write me a joke"))
#
#     # Define benchmark with n_problems and shots
#     benchmark = GSM8K(
#         n_problems=100,
#         n_shots=0,
#         enable_cot=False
#     )
#
#     # Replace 'mistral_7b' with your own custom model
#     benchmark.evaluate(model=openai)
#     print(benchmark.overall_score)

if __name__ == '__main__':
    class OpenAI(DeepEvalBaseLLM):
        def __init__(self):
            self.solver = Gsm8kGraph(name="solver", llm=LLM())

        def load_model(self):
            # 这里应该是加载模型的逻辑
            pass

        async def a_generate(self, prompt: str) -> str:
            solution_result = await self.solver(prompt)
            return solution_result['solution']

        def generate(self, prompt: str) -> str:
            loop = asyncio.get_event_loop()
            solution_result = loop.run_until_complete(self.a_generate(prompt))  # 等待 a_generate 方法完成
            return solution_result

        def get_model_name(self):
            return "Custom Azure OpenAI Model"

    # Replace these with real values
    openai = OpenAI()
    # print(openai.generate("Write me a joke"))

    # Define benchmark with n_problems and shots
    benchmark = GSM8K(
        n_problems=10,
        n_shots=0,
        enable_cot=False
    )

    async def async_evaluate_problem(model, golden, benchmark):
        prompt = GSM8KTemplate.generate_output(
            train_set=benchmark.shots_dataset,
            input=golden.input,
            n_shots=benchmark.n_shots,
            enable_cot=benchmark.enable_cot,
        )
        prediction = await model.a_generate(prompt)
        score = benchmark.scorer.exact_match_score(golden.expected_output, prediction)
        return golden.input, prediction, golden.expected_output, score


    async def evaluate_benchmark(benchmark, model):
        goldens = benchmark.load_benchmark_dataset()[:benchmark.n_problems]
        tasks = [async_evaluate_problem(model, golden, benchmark) for golden in goldens]
        results = await asyncio.gather(*tasks)

        overall_correct_predictions = sum(score for _, _, _, score in results)
        overall_total_predictions = benchmark.n_problems
        overall_accuracy = overall_correct_predictions / overall_total_predictions

        predictions_row = [(input, prediction, expected_output, score) for input, prediction, expected_output, score in
                           results]
        benchmark.predictions = pd.DataFrame(predictions_row,
                                             columns=["Input", "Prediction", "Expected Output", "Correct"])
        benchmark.overall_score = overall_accuracy

        now = datetime.datetime.now().time()
        now_time = now.strftime("%Y-%m-%d_%H-%M-%S").replace(':', '_')

        # Save the detailed data to a CSV file
        benchmark.predictions.to_csv(f'gsm8k_{overall_accuracy}_{now_time}.csv', index=False)

        print(f"Overall GSM8K Accuracy: {overall_accuracy}")
        return overall_accuracy


    loop = asyncio.get_event_loop()
    loop.run_until_complete(evaluate_benchmark(benchmark, openai))

    # Print the overall score
    print(benchmark.overall_score)
