from typing import Literal, List, Dict, Tuple
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
import random
from collections import Counter
from metagpt.llm import LLM
from metagpt.provider.llm_provider_registry import create_llm_instance
from examples.ags.w_action_node.operator import Operator
from metagpt.actions.action_node import ActionNode
from examples.ags.w_action_node.optimized.Gsm8K.operators.template.operator_an import *
from examples.ags.w_action_node.optimized.Gsm8K.operators.Generate.round_1.prompt import *

class Generate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(name, llm)

    async def __call__(self, problem):
        prompt = GENERATE_PROMPT.format(problem=problem)
        node = await ActionNode.from_pydantic(GenerateOp).fill(context=prompt, llm=self.llm, mode="single_fill")
        response = node.instruct_content.model_dump()
        return response
                    