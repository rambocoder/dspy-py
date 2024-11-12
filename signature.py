from dotenv import load_dotenv
from pprint import pprint
import os
load_dotenv()
import dspy

lm = dspy.LM('openai/gpt-4o-mini', temperature=0.1) # openai/gpt-3.5-turbo
dspy.configure(lm=lm)

sentence = "A nice and sunny day in Hawaii."  
classify = dspy.Predict('sentence -> sentiment')
sentiment = classify(sentence=sentence).sentiment
pprint(sentiment)

# question -> answer" or "document -> summary" 

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

generate_answer = dspy.ChainOfThought(GenerateSearchQuery)

answer =generate_answer(question="What is the weather like in Hawaii?", context="A nice and sunny day in Hawaii.")
print(answer.query)

cot = dspy.ChainOfThought('question -> answer')
pprint(cot(question="What is 2+2?"))


fact_checking = dspy.ChainOfThought('claims -> verdicts: list[bool]')
pprint(fact_checking(claims=["Python was released in 1991.", "Python is a compiled language."]))

dspy.inspect_history()

