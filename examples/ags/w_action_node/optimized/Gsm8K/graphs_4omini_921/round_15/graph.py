from typing import Literal
import examples.ags.w_action_node.optimized.Gsm8K.graphs.template.operator as operator
import examples.ags.w_action_node.optimized.Gsm8K.graphs.round_15.prompt as prompt_custom
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
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        analysis = await self.custom(input=problem, instruction=prompt_custom.ANALYZE_PROMPT)
        code_result = await self.programmer(problem=problem, analysis=analysis['response'])
        
        solutions = []
        for approach in ['SOLVE_PROMPT', 'SOLVE_PROMPT_ALT', 'SOLVE_PROMPT_NUMERIC']:
            solution = await self.custom(input=problem + f"\nAnalysis: {analysis['response']}\nCode output: {code_result['output']}", instruction=getattr(prompt_custom, approach))
            solutions.append(solution['response'])
        
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        double_check = await self.custom(input=problem + f"\nBest solution: {best_solution['response']}", instruction=prompt_custom.DOUBLE_CHECK_PROMPT)
        final_solution = await self.custom(input=problem + f"\nBest solution: {best_solution['response']}\nDouble-check: {double_check['response']}", instruction=prompt_custom.FINAL_SOLUTION_PROMPT)
        return final_solution['response'], self.llm.cost_manager.total_cost
                    