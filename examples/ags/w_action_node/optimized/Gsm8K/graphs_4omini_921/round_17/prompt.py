ANALYZE_PROMPT = """
Analyze the given math problem. Identify the key information, variables, and operations needed to solve it. Provide a step-by-step approach to solving the problem, but do not perform any calculations.

Problem:
{input}

Provide your analysis:
"""

SOLVE_PROMPT_1 = """
Solve the given math problem using the provided analysis. Ensure your solution is clear, concise, and provides the final answer in a format that can be easily extracted (e.g., "Final answer: X"). Use a direct calculation approach.

Problem:
{input}

Provide your solution:
"""

SOLVE_PROMPT_2 = """
Solve the given math problem using the provided analysis. Ensure your solution is clear, concise, and provides the final answer in a format that can be easily extracted (e.g., "Final answer: X"). Use an alternative method or perspective if possible.

Problem:
{input}

Provide your solution:
"""

REFINE_PROMPT = """
Compare and refine the two solutions provided for the given math problem. Consider the code output for verification. Provide a step-by-step explanation of your refinement process and the resulting solution.

Problem:
{input}

Provide your refined solution:
"""

DOUBLE_CHECK_PROMPT = """
Double-check the refined solution for the given math problem. If possible, use a different method or perspective to verify the answer. Provide a step-by-step explanation of your verification process and the result.

Problem:
{input}

Provide your double-check solution:
"""

FINAL_SOLUTION_PROMPT = """
Compare the refined solution and the double-check solution. Determine the final answer based on this comparison. If there's a discrepancy, explain why you chose one answer over the other. Provide the final answer in a clear, extractable format (e.g., "Final answer: X").

Problem:
{input}

Provide your final solution:
"""