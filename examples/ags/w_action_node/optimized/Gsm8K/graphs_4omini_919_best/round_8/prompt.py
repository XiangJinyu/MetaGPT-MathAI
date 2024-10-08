SOLVE_PROMPT = """
Solve the given math problem step by step. Show your work and clearly state the final answer.

1. Read the problem carefully and identify the key information.
2. Determine the appropriate mathematical operations needed to solve the problem.
3. Perform the calculations step by step, showing your work.
4. Clearly state the final answer, including the appropriate units if applicable.
5. Double-check your calculations for accuracy.

Provide your solution below:
"""

ALTERNATIVE_SOLVE_PROMPT = """
Solve the given math problem using a different approach or method than you might typically use. Show your work step by step and clearly state the final answer.

1. Read the problem carefully and identify the key information.
2. Think of an alternative method or approach to solve the problem.
3. Perform the calculations step by step, showing your work.
4. Clearly state the final answer, including the appropriate units if applicable.
5. Verify your solution by checking if it makes sense in the context of the problem.

Provide your alternative solution below:
"""

REVIEW_PROMPT = """
Review the given problem and solution. Verify the accuracy of the solution and make corrections if necessary. If the solution is correct, simply restate the final answer. If there are errors, provide the correct solution with explanations.

1. Read the original problem and the provided solution.
2. Check each step of the solution for accuracy.
3. Verify that the final answer is correct and appropriate for the given problem.
4. If there are any errors, provide the correct solution with clear explanations.
5. If the solution is correct, restate the final answer.

Provide your review and final answer below:
"""

ENSEMBLE_PROMPT = """
Compare and analyze the two given solutions for the same problem. Determine which solution is more accurate or if they agree. If they differ, explain why and provide a final, correct answer.

1. Carefully read both solutions.
2. Compare the approaches and results of both solutions.
3. If the solutions agree, confirm the final answer.
4. If the solutions disagree, analyze the discrepancies and determine the correct approach.
5. Provide a final, verified answer with a brief explanation of your reasoning.

Provide your analysis and final answer below:
"""

VERIFY_PROMPT = """
Verify the ensemble solution for the given problem. Double-check all calculations and reasoning. If any errors are found, provide a corrected solution with explanations.

1. Carefully read the original problem and the ensemble solution.
2. Verify each step of the solution, including all calculations and logical reasoning.
3. Check if the final answer makes sense in the context of the problem.
4. If any errors are found, provide a detailed explanation and the correct solution.
5. If the solution is correct, confirm the final answer and provide a brief explanation of why it's correct.

Provide your verification and final answer below:
"""

CONTEXT_CHECK_PROMPT = """
Review the original problem and the final solution. Check if the answer makes sense in the context of the problem. If there are any inconsistencies or errors, provide a corrected answer with explanations.

1. Carefully read the original problem and the final solution.
2. Analyze whether the final answer aligns with the problem's context and requirements.
3. Check if the units, scale, and type of the answer are appropriate for the question.
4. If any inconsistencies are found, recalculate and provide the correct answer with a detailed explanation.
5. If the answer is correct and consistent with the problem context, confirm it and briefly explain why it fits.

Provide your context check and final answer below:
"""

EXTRACT_ANSWER_PROMPT = """
Extract only the final numerical answer from the given solution. If there are multiple numbers, choose the one that represents the final answer to the problem. Do not include any units or additional text. Provide only the number as a decimal or integer.

For example, if the solution contains "The final answer is $20.00", your response should be:
20.00

Extract and provide only the numerical answer below:
"""