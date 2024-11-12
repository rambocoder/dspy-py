import dspy
from dspy.evaluate import Evaluate
from dspy import Example
from dspy.teleprompt import BootstrapFewShot  # Add this import

from dotenv import load_dotenv
load_dotenv()

lm = dspy.LM('openai/gpt-4o-mini', temperature=0.1) # openai/gpt-3.5-turbo
dspy.configure(lm=lm)

# Define the initial prompt and the correct answer
initial_prompt = "What is 2+2?"
correct_answer = "4"

# Create a few-shot learning instance using BootstrapFewShot
# This will optimize the prompt based on the examples provided
few_shot_optimizer = BootstrapFewShot(
    examples=[
        Example(input="What is 2+2?", output="4"),
        Example(input="What is 3+3?", output="6"),
        Example(input="What is 5+5?", output="10")
    ],
    max_iterations=5  # Number of iterations to optimize the prompt
)

# Run the optimization process
optimized_prompt = few_shot_optimizer.compile(initial_prompt)

# Test the optimized prompt with the model
response = few_shot_optimizer.generate(optimized_prompt)

# Print the optimized prompt and the model's response
print(f"Optimized Prompt: {optimized_prompt}")
print(f"Model's Response: {response}")