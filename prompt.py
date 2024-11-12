import dspy
from dspy.evaluate import Evaluate
from dspy import ChainOfThought, Example
from dspy.teleprompt import BootstrapFewShot  # Add this import

from dotenv import load_dotenv
load_dotenv()

lm = dspy.LM('openai/gpt-3.5-turbo', temperature=0.1) # openai/gpt-3.5-turbo gpt-4o-mini
dspy.configure(lm=lm)

# Define our basic prompt signature
class SimpleMath(dspy.Signature):
    """Answer simple math questions."""
    
    question = dspy.InputField(desc="The math question")
    answer = dspy.OutputField(desc="The numerical answer")

# Create our program using ChainOfThought
class MathSolver(dspy.Program):
    def __init__(self):
        super().__init__()
        self.solver = ChainOfThought(SimpleMath)
    
    def forward(self, question):
        return self.solver(question=question)

# Create training data
train_data = [
        Example(question="T:: What is 2+2?", answer="4.00").with_inputs("question"),
        Example(question="T:: What is 3+3?", answer="6.00").with_inputs("question"),
        Example(question="T:: What is 5+5?", answer="10.00").with_inputs("question"),
        Example(question="T:: What is five minus five?", answer="0.00").with_inputs("question")
    ]

# Create validation data
valid_data = [
        Example(question="V:: What is two plus two?", answer="4.00").with_inputs("question"),
        Example(question="V:: What is 10 - 8?", answer="2.00").with_inputs("question"),
        Example(question="V:: Add one to one", answer="2.00").with_inputs("question")
    ]


def validate_answer(example, pred, trace=True):
    return example.answer.lower() == pred.answer.lower()

# Create evaluator
evaluator = Evaluate(
    metric=validate_answer,
    devset=valid_data
)



# Manually adjust the prompt (example)
# This is a placeholder for any manual optimization you might do
# For example, you could adjust the temperature, prompt structure, etc.

# Test the program
# question = "what is 2+2?"
# program = MathSolver()
# result = program(question=question)
# print(f"Question: {question}")
# print(f"Answer: {result.answer}")

# # Evaluate the program
# metrics = evaluator(program)
# print(f"Evaluation metrics: {metrics}")

# exit(0)

# Optimize the prompt
# Use BootstrapFewShot instead of OptimizePrompt
optimizer = BootstrapFewShot(
    metric=validate_answer
)

# Compile the program using the optimizer
trained_program = optimizer.compile(
    MathSolver(),
    trainset=train_data,
    # valset=valid_data
)

# dspy.inspect_history()

# Test the optimized program
question = "One hundred plus two equals = "
result = trained_program(question=question)
print(f"Question: {question}")
print(f"Answer: {result.answer}")

trained_program.save('trained.dat')

# import pickle
# with open('optimizer.pkl', 'wb') as f:
#   pickle.dump(trained_program, f)


# Evaluate the program
metrics = evaluator(trained_program)
print(f"Evaluation metrics: {metrics}")

# dspy.inspect_history()


loaded_program = MathSolver()
loaded_program.load(path="trained.dat")
question = "Joe had five apples and Jim gave him two more apples, how apples does Joe have now?"
result = loaded_program(question=question)
print(f"Question 2: {question}")
print(f"Answer 2: {result.answer}")