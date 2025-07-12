from langgraph.graph import StateGraph, add_messages, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(BaseModel):
    messages: Annotated[List, add_messages]


def build_agent_graph(tools: List[BaseTool] = []):

    system_prompt = """
    Your name is Data Alchemist and you are an expert data scientist. You help users analyze data by leveraging tools
    available to you. Your job is to collaborate with the user to help build their understanding of the data
    
    <filesystem>
    You have access to a set of tools that allow you to interact with the user's local filesystem.
    You are only allowed to access files within the working directory 'data'.
    The path to this directory is {working_dir}
    </filesystem
    
    <data>
    When the user refers to the data for the project, they are referring to data within the working directory.
    You can provide descriptive information about this data.
    You have a set of tools for data inspection and cleaning that will allow you to perform imputation
    for missing values, perform encoding for categorical features, and drop (remove) specified columns.
    </data
    
    <modeling>
    You also have access to tools that allow you to split data into training and testing sets
    and a tool for performing linear regression. 
    </modeling>
    
    <tools>
    {tools}
    </tools>
    
    Assist the user in all aspects of the data analysis.
    """

    llm = ChatGoogleGenerativeAI(name="Data Alchemist", model="gemini-2.5-flash")
    if tools:
        llm = llm.bind(tools=tools)
        # inject tools into system prompt
        tools_json = [tool.model_dump_json(include=['name', 'description']) for tool in tools]

    def assistant(state: AgentState) -> AgentState:
        response = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
        state.messages.append(response)
        return state

    builder = StateGraph(AgentState)

    builder.add_node("Data_Alchemist", assistant)
    builder.add_node(ToolNode(tools))

    builder.add_edge(START, "Data_Alchemist")
    builder.add_conditional_edges(
        "Data_Alchemist",
        tools_condition,
    )

    builder.add_edge("tools", "Data_Alchemist")

    return builder.compile(checkpointer=MemorySaver())


# visualize graph
if __name__ == "__main__":
    from IPython.display import display, Image

    graph = build_agent_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))

