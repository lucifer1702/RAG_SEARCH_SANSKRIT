from typing import List, Union

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.function_calling.step import FunctionCallingAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.program.function_program import _get_function_tool
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

# Pydantic defintions
class Step(BaseModel):
    """A single step in a plan."""
    
    name: str = Field(..., description="The name of the step.")
    input: str = Field(..., description="The input prompt for the step.")
    expected_output: str = Field(..., description="The expected output of the step.")
    dependencies: List[str] = Field(..., description="The step names that must be completed before this step.")

class Plan(BaseModel):
    """A series of steps to accomplish a task."""

    steps: List[Step] = Field(..., description="The steps in the plan.")


# Load documents, create tools
lyft_documents = SimpleDirectoryReader(input_files=["./docs/docs/examples/data/10k/lyft_2021.pdf"]).load_data()
uber_documents = SimpleDirectoryReader(input_files=["./docs/docs/examples/data/10k/uber_2021.pdf"]).load_data()

lyft_index = VectorStoreIndex.from_documents(lyft_documents)
uber_index = VectorStoreIndex.from_documents(uber_documents)

lyft_tool = QueryEngineTool.from_defaults(
    lyft_index.as_query_engine(),
    name="lyft_2021",
    description="Useful for asking questions about Lyft's 2021 10-K filling.",
)

uber_tool = QueryEngineTool.from_defaults(
    uber_index.as_query_engine(),
    name="uber_2021",
    description="Useful for asking questions about Uber's 2021 10-K filling.",
)


# create planning prompt
tools = [lyft_tool, uber_tool]
tools_str = ""
for tool in tools:
    tools_str += tool.metadata.name + ": " + tool.metadata.description + "\n"


task = "Summarize the key risk factors for Lyft and Uber in their 2021 10-K filings."

prompt = f"""\
Think step-by-step. Given a task and a set of tools, create a comprehesive, end-to-end plan to accomplish the task. 
Keep in mind not every task needs to be decomposed into multiple steps if it is simple enough. 
Be sure to group parallel tasks with the ParallelStep class.
The plan should end with a step that satisfies the original task.

The tools available are:
{tools_str}

Overall Task: {task}
"""

# generate the plan
llm = OpenAI(model="gpt-4-turbo")
plan_tool = _get_function_tool(Plan)

response = llm.chat_with_tools([plan_tool], user_msg=prompt, allow_parallel_tool_calls=False)
tool_choice = llm.get_tool_calls_from_response(response)
plan = Plan(**tool_choice[0].tool_kwargs)


def get_next_steps(plan: Plan, completed_steps: List[str]) -> List[Step]:
    next_steps = []
    for step in plan.steps:
        if step.name not in completed_steps and all(dep in completed_steps for dep in step.dependencies):
            next_steps.append(step)
    return next_steps

# execute the plan
agent_worker = FunctionCallingAgentWorker.from_tools(tools, llm=llm)
agent = AgentRunner(agent_worker)

completed_steps = []
while len(completed_steps) < len(plan.steps):
    next_steps = get_next_steps(plan, completed_steps)

    for step in next_steps:
        response = agent.chat(step.input)
        completed_steps.append(step.name)

print(str(response))

