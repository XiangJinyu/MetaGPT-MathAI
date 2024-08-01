# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 22:07 PM
# @Author  : didi
# @Desc    : graph & an instance - humanevalgraph

from metagpt.llm import LLM 
from typing import List
from examples.ags.w_action_node.operator import Generate, GenerateCode, GenerateCodeBlock, Review, Revise, FuEnsemble, MdEnsemble, DbEnsemble, Rephrase, Test
from examples.ags.w_action_node.utils import extract_test_cases_from_jsonl
from evalplus.data import get_human_eval_plus
class Graph:
    def __init__(self, name:str, llm:LLM) -> None:
        self.name = name
        self.model = llm 

    def __call__():
        NotImplementedError("Subclasses must implement __call__ method")

    def optimize(dataset:List):
        pass

class HumanEvalGraph(Graph):
    def __init__(self, name:str, llm: LLM, criteria:str, vote_count:int =5) -> None:
        super().__init__(name, llm)
        self.criteria = criteria # TODO 自动构建图时，图的初始参数与图所使用的算子要求的外部参数相匹配
        self.generate_code = GenerateCode(llm=llm)
        self.generate_code_block = GenerateCodeBlock(llm=llm)
        self.review = Review(llm=llm, criteria=criteria)
        self.revise = Revise(llm=llm)
        self.rephrase = Rephrase(llm=llm)
        self.tester = Test(llm=llm)
        self.fuensemble = FuEnsemble(llm=llm)
        self.mdensemble = MdEnsemble(llm=llm, vote_count=vote_count)

    async def __call__(self, problem:str, ensemble_count:int = 3):
        solution_list = []
        for _ in range(ensemble_count):
            for retry_count in range(5):
                try:
                    # solution = await self.generate_code(problem)
                    solution = await self.generate_code_block(problem)
                    solution = solution.get('code_solution')
                    solution_list.append(solution)
                    break
                except Exception as e:
                    print(e)
        solution = await self.mdensemble("code", solution_list, problem)
        return solution
    
    async def alpha_codium(self, problem_id:str, problem:str, ensemble_count:int = 3):
    # async def __call__(self,problem_id, problem:str, ensemble_count:int = 3):
        test_cases = extract_test_cases_from_jsonl(problem_id)
        entry_point = get_human_eval_plus()[problem_id]['entry_point']
        rephrase_problem = await self.rephrase(problem) # 在rephrase 中拼接原始的问题描述
        solution_list = []
        for _ in range(ensemble_count):
            for retry_count in range(5):
                try:
                    solution = await self.generate_code_block.rephrase_generate(problem, rephrase_problem, function_name=entry_point)
                    solution = solution.get('code_solution')
                    solution_list.append(solution)
                    break
                except Exception as e:
                    print(e)
        solution = await self.mdensemble("code", solution_list, problem)
        solution = await self.tester(problem_id, problem, rephrase_problem, solution, test_cases)
        return solution

    async def review_revise_ensemble(self, problem:str, ensemble_count:int = 2):
        solution_list = []
        for _ in range(ensemble_count):
            solution = await self.single_solve(problem, 3)
            solution_list.append(solution)
        solution = await self.ensemble(solution_list, problem)
        return solution

    async def simple_ensemble(self, problem:str, ensemble_count:int = 3):
    # async def __call__(self, problem:str, ensemble_count:int = 3):
        solution_list = []
        for _ in range(ensemble_count):
            solution = await self.generate_code(problem)
            # solution = await self.generate_code_block(problem)
            solution = solution.get('code_solution')
            solution_list.append(solution)
        solution = await self.fuensemble(solution_list, problem)
        return solution
    
    async def single_solve(self, problem:str, max_loop:int):
        solution = await self.generate_code(problem)
        solution = solution.get('code_solution')
        for _ in range(max_loop):
            review_feedback = await self.review(problem, solution)
            if review_feedback['review_result']:
                break
            solution = await self.revise(problem, solution, review_feedback['feedback'])
            solution = solution.get('revised_solution')
        return solution

    
    # async def __call__(self, problem:str):
    # 这个地方没有修改对应的prompt，可以对应着humaneval改一下
    #     problem = await self.rephrase(problem)
    #     solution = self.generate(problem)
    #     return solution