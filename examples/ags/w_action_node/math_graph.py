# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 22:07 PM
# @Author  : didi
# @Desc    : graph & an instance - humanevalgraph

from metagpt.llm import LLM
from typing import List
from examples.ags.w_action_node.math_operator import Generate, Rephrase, Format
from examples.ags.w_action_node.utils import extract_test_cases_from_jsonl


class Graph:
    def __init__(self, name: str, llm: LLM) -> None:
        self.name = name
        self.model = llm

    def __call__():
        NotImplementedError("Subclasses must implement __call__ method")

    def optimize(dataset: List):
        pass


class Gsm8kGraph(Graph):
    def __init__(self, name: str, llm: LLM) -> None:
        super().__init__(name, llm)
        self.generate = Generate(llm=llm)
        self.rephrase = Rephrase(llm=llm)
        self.format = Format(llm=llm)

    async def __call__(self, problem: str):
        formatted_problem = await self.rephrase(problem)
        solution = await self.generate(formatted_problem)  # 确保等待 generate 方法完成
        solution0 = solution['solution']
        formatted_solution = await self.format(solution0)  # 确保等待 generate 方法完成
        return formatted_solution

    # async def __call__(self, problem:str):
    # 这个地方没有修改对应的prompt，可以对应着humaneval改一下
    #     problem = await self.rephrase(problem)
    #     solution = self.generate(problem)
    #     return solution