from typing import Literal
import examples.ags.w_action_node.optimized.Gsm8K.graphs.template.operator as operator
import examples.ags.w_action_node.optimized.Gsm8K.graphs.round_16.prompt as prompt_custom
import examples.ags.w_action_node.optimized.Gsm8K.graphs.template.prompt_lib as prompt_lib
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
        self.generate = operator.Generate(self.llm)
        self.format = operator.Format(self.llm)
        self.rephrase = operator.Rephrase(self.llm)
        self.review = operator.Review(self.llm)
        self.revise = operator.Revise(self.llm)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        rephrased_problem = await self.rephrase(problem=problem)
        solution = await self.generate(problem=rephrased_problem['response'])
        review_result = await self.review(problem=problem, solution=solution['response'])
        if not review_result['review_result']:
            solution = await self.revise(problem=problem, solution=solution['response'], feedback=review_result['feedback'])
        enhanced_solution = await self.custom(input=f"Problem: {problem}\nSolution: {solution['response']}", instruction=prompt_custom.ENHANCE_SOLUTION)
        
        # Compare enhanced solution with original solution
        comparison = await self.custom(input=f"Original solution: {solution['response']}\nEnhanced solution: {enhanced_solution['response']}", instruction=prompt_custom.COMPARE_SOLUTIONS)
        
        if comparison['response'] == "similar":
            creative_solution = await self.custom(input=f"Problem: {problem}\nPrevious solution: {enhanced_solution['response']}", instruction=prompt_custom.CREATIVE_SOLUTION)
            final_solution = creative_solution['response']
        else:
            final_solution = enhanced_solution['response']
        
        format_solution = await self.format(problem=problem, solution=final_solution)
        return format_solution['response'], self.llm.cost_manager.total_cost
                    