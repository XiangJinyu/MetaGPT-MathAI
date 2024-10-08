from typing import Literal
from examples.ags.w_action_node.optimized.Gsm8K.graphs.template.operator import *
from examples.ags.w_action_node.optimized.Gsm8K.graphs.round_5.prompt import *
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
        self.format = Format(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        question = await self.generate(input=problem, prompt=REPHRASE_PROMPT)
        solution = await self.generate(input=question['content'], prompt=GENERATE_PROMPT)
        self_question = await self.generate(input=f"Problem: {problem}\n\nSolution: {solution['content']}", prompt=SELF_QUESTION_PROMPT)
        if "revise" in self_question['content'].lower():
            solution = await self.generate(input=f"Problem: {problem}\n\nOriginal solution: {solution['content']}\n\nSelf-questioning feedback: {self_question['content']}", prompt=GENERATE_PROMPT)
        review = await self.generate(input=f"Problem: {problem}\n\nSolution: {solution['content']}", prompt=REVIEW_PROMPT)
        if "issues are found" in review['content'].lower():
            solution = await self.generate(input=f"Problem: {problem}\n\nOriginal solution: {solution['content']}\n\nReview feedback: {review['content']}", prompt=GENERATE_PROMPT)
        format_solution = await self.format(input=f"Original question:{problem} \n\nFinal solution:{solution['content']}")
        return format_solution, self.llm.cost_manager.total_cost
                    