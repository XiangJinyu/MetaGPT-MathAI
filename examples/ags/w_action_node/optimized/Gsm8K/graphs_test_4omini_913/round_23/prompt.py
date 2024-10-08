ANALYZE_DIFFICULTY = """
Analyze the given mathematical problem and determine its difficulty level. Consider the following factors:
1. The complexity of mathematical concepts involved.
2. The number of steps likely required to solve the problem.
3. The presence of any advanced or specialized topics.
4. The level of abstract thinking required.

Based on your analysis, classify the problem as either "low", "medium", or "high" difficulty. Provide only the difficulty level as your response, without any additional explanation.
"""

REVISION_CONTEXT = """
Given the problem and the original solution, provide additional context or insights that might be helpful for revising the solution. Consider:
1. Any important mathematical concepts or formulas that might have been overlooked.
2. Potential alternative approaches to solving the problem.
3. Common pitfalls or mistakes related to this type of problem.
4. Any relevant real-world applications or examples that could enhance understanding.

Provide this context in a concise, clear manner to assist in improving the solution.
"""

ALTERNATIVE_APPROACH = """
Given the mathematical problem, its visual representation, and difficulty level, suggest an alternative approach to solving it. This approach should be different from the most obvious or standard method. Consider:
1. Using a different mathematical technique or concept.
2. Approaching the problem from a unique angle or perspective.
3. Applying a less common but potentially more efficient method.
4. Utilizing the provided visual representation to gain new insights.
5. Adjusting the complexity of the approach based on the difficulty level.

Provide a brief description of this alternative approach, focusing on its key steps or principles.
"""

VISUAL_REPRESENTATION = """
Given the mathematical problem, create a concise textual description of a visual representation or diagram that could help in understanding and solving the problem. Consider:
1. Key elements of the problem that can be visualized.
2. Relationships between different components of the problem.
3. Any geometric shapes, graphs, or charts that might be relevant.
4. How the visual representation could highlight important aspects of the problem.

Provide a clear and detailed description of the visual representation, focusing on how it relates to the problem and could aid in finding a solution.
"""