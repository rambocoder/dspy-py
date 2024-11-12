# Import necessary modules from DSPy
from dspy import BootstrapFewShot

# Load the optimized prompter from the file
loaded_prompter = BootstrapFewShot.compile()

# Test the loaded prompter by generating a response
response = loaded_prompter.generate("What is 2 plus 2?")

# Print the model's response
print(f"Model's Response: {response}")


question = "One hundred plus two equals = "
result = trained_program(question=question)
print(f"Question: {question}")
print(f"Answer: {result.answer}")