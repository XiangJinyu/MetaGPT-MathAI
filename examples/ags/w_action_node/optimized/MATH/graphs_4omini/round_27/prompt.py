PRIME_FACTORIZATION_PROMPT = """
Given the problem statement, identify the target product value and perform a prime factorization of this value. If the problem doesn't explicitly mention a product value, infer it from the context. Present the prime factors in a clear, organized manner.

For example, if the target product value is 60, the prime factorization would be:
60 = 2 * 2 * 3 * 5

Provide only the prime factorization result without any additional explanation.
"""

GENERATE_SOLUTION_APPROACHES = """
Based on the problem statement, ensemble result, and prime factorization, generate multiple solution approaches to solve the problem. Each approach should be distinct and offer a unique perspective on solving the problem. Present at least three different solution strategies, explaining the rationale behind each approach.

For each approach:
1. Provide a brief description of the strategy.
2. Outline the key steps involved.
3. Explain how it utilizes the given information (prime factors, ensemble result).

Present your solution approaches in a clear, numbered format.
"""

REFINE_SOLUTION_PROMPT = """
Review the given problem, the ensemble result of generated code solutions, the prime factorization, and the multiple solution approaches. Provide a refined, step-by-step explanation of the solution, incorporating the best elements from the various approaches. Ensure all calculations are correct and the final answer is clearly stated. If there are any errors in the previous solutions, identify and correct them. Your response should be a complete, mathematically rigorous solution to the problem.

Important: Format your final answer using LaTeX notation enclosed in \boxed{}, for example: \boxed{42} or \boxed{x + y}.
"""

ERROR_CHECK_PROMPT = """
Carefully review the problem statement and the refined solution. Check for any logical errors, calculation mistakes, or inconsistencies in the solution. If you find any errors, provide a corrected solution. If the solution is correct, simply confirm its accuracy.

Ensure that:
1. All calculations are mathematically correct.
2. The solution directly addresses the problem statement.
3. The final answer is clearly stated and formatted using LaTeX notation enclosed in \boxed{}.
4. The units of measurement are correct and consistent throughout the solution.
5. Any assumptions made are valid and clearly stated.

If corrections are needed, provide a complete, step-by-step solution. If the original solution is correct, state that it is accurate and restate the final answer.
"""