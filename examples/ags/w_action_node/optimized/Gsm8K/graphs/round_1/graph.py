# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 22:07 PM
# @Author  : didi
# @Desc    : Basic Graph Class

from typing import Literal

from examples.ags.w_action_node.optimized.Gsm8K.graphs.round_1.operator import *
from examples.ags.w_action_node.optimized.Gsm8K.graphs.round_1.prompt import *
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MMBP", "Gsm8K", "MATH", "HotpotQa", "MMLU"]

class SolveGraph:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.generate = Generate(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        solution = await self.generate(problem, prompt=GENERATE_PROMPT)
        return solution, self.llm.cost_manager.total_cost
