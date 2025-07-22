"""
Multi-Agent LangGraph with Speech-to-Text Node (Swapfiets)
"""
import os
import sys
from typing import Literal
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import init_chat_model
from langgraph.graph import StateGraph, END, MessagesState
from langchain.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.graph.schema import ChatMessage, State
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Optional
import openai

# Load environment variables
load_dotenv()


# 1. State definition
class State(MessagesState):
    current_agent: Literal[
        "speech_to_text",
        "issue_classifier",
        "clarification",
        "repair_diagnostics",
        "repair_vs_replace"
    ]


# 2. Tool: Speech-to-text using Whisper

# Define tools and agents
class SpeechToTextTool(BaseTool):
    name: str = "speech_to_text"
    description: str = "Transcribes an audio file to text using Whisper."

    def _run(self, file_path: str) -> str:
        with open(file_path, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        return response["text"]

    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)

speech_to_text_tool = SpeechToTextTool()

# Model initialization
llm = ChatOpenAI(model="gpt-4")

# Pydantic schema for classification
class IssueClassification(BaseModel):
    bike_type: Optional[str]
    part_category: Optional[str]
    part_name: Optional[str]
    position: Optional[str]
    issue: Optional[str]
    likely_service: Optional[str]


# 3. Node: Speech-to-text agent

def speech_to_text_node(state: State) -> State:
    audio_path = next(
        (msg.content for msg in state.messages if isinstance(msg, HumanMessage) and msg.content.endswith(".mp3")), None)
    if not audio_path:
        raise ValueError("No audio file path found in message content")

    transcript = speech_to_text_tool.run(audio_path)
    new_messages = state.messages + [AIMessage(content=transcript)]
    return State(messages=new_messages, current_agent="issue_classifier")


# Agent: Issue classifier
def issue_classifier_node(state: State) -> State:
    last_text = next((msg.content for msg in reversed(state.messages) if isinstance(msg, AIMessage)), None)
    if not last_text:
        raise ValueError("No transcribed input available.")

    prompt = f"""
    You are an expert mechanic at Swapfiets. Extract the following attributes from the problem described:
    - bike_type
    - part_category
    - part_name
    - position (front, rear, left, right, or null)
    - issue
    - likely_service (repair, replace, adjust, lubricate, tighten)

    Respond in JSON format like:
    {{
        "bike_type": "Deluxe 7",
        "part_category": "Drivetrain",
        "part_name": "chain",
        "position": "rear",
        "issue": "slipping",
        "likely_service": "replace"
    }}

    Problem: {last_text}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content
    return State(messages=state.messages + [AIMessage(content=classification)], current_agent="repair_diagnostics")


# 4. Node: Router logic

def router(state: State) -> State:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            query = message.text()
            break

    model = init_chat_model(provider="openai")

    messages = [
        SystemMessage(content="""
        You are a router for a Swapfiets customer service system. Your job is to determine what is wrong with a bike based on the member's feedback.

        The available agents are:
        1. speech_to_text
        2. issue_classifier
        3. clarification
        4. repair_diagnostics
        5. repair_vs_replace
        """),
        HumanMessage(content=query)
    ]
    response = model.invoke(messages)

    content = response.content.lower()
    if "audio" in content or "spoken" in content:
        agent = "speech_to_text"
    # elif "clarify" in content:
    #     agent = "clarification"
    # elif "replace" in content:
    #     agent = "repair_vs_replace"
    # elif "repair" in content or "diagnose" in content:
    #     agent = "repair_diagnostics"
    # else:
    #     agent = "issue_classifier"

    return State(messages=state.messages, current_agent=agent)


# 5. Build LangGraph

graph = StateGraph(State)

# Add nodes
graph.add_node("router", router)
graph.add_node("speech_to_text", speech_to_text_node)
graph.add_node("issue_classifier", issue_classifier_node)

# Future nodes would go here:
# graph.add_node("clarification", clarification_node)
# graph.add_node("repair_diagnostics", repair_diagnostics_node)
# graph.add_node("repair_vs_replace", repair_vs_replace_node)

# Add edges
graph.set_entry_point("router")
graph.add_edge("router", lambda state: state.current_agent)
graph.add_edge("speech_to_text", END)

# Terminate at placeholder for now
# graph.add_edge("speech_to_text", END)

app = graph.compile()
