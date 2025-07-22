import os
import sys
from typing import Literal
from dotenv import load_dotenv
from utils.model_utils import init_chat_model
from utils.speech_to_text_tool import SpeechToTextTool
from utils.issue_classifier import IssueClassification
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

import openai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


speech_to_text_tool = SpeechToTextTool()
parser = PydanticOutputParser(pydantic_object=IssueClassification)

llm = ChatOpenAI(model="gpt-4")

sys_msg = SystemMessage(content="You are a helpful assistant trying to understand what is wrong with the member's bike. If the issue_classifier_node does not have enough information, ask the member for more feedback")

def speech_to_text_node(state: State) -> State:
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


# Agent: Issue classifier
def issue_classifier_node(state: State) -> State:
    # Grab the last AI message (from speech_to_text)
    last_msg = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)),
        None
    )

    if not last_msg:
        raise ValueError("No message found for issue classification")


    prompt = f"""
    You are a technical classification assistant. Your job is to extract structured information from a customer's natural language description of a bike problem. If the information provided is not enough to build the JSON then redirect back to the speech_to_text agent so it can ask follow-up questions:
    - bike_type: it needs to be one of these - Deluxe 7, Original 1, Original 1+, Power 1, Power 7 or Power Plus
    - part_category: it needs to be one of these - Brakes, Drivetrain, Fenders, Frame, Gears, Handlebar, Light, Lock, Saddle, Wheel, Electrics, Folding, Body & Panel
    - part_name: it needs to be one of these - Brake cable/hose, Brake lever, Brake pads, Brake unit/caliper, Bottom bracket, Chain, Chain wheel, Chainguard, Crank, Pedals, Sprocket, Fender, Fender stay, Barcode, Carrier, Carrier bracket, Carrier bumper, Frame/panel, Front fork, Kickstand, Kickstand foot, Cassette-joint, Gears, Shifter, Shifter cable, Bell, Grips, Handlebar, Headset, Headset cover, Stem, Bye bye battery, Cable, Light, Magnet, Reflector, Chain lock, Frame lock, Saddle, Saddle clamp, Seatpost, Seatpost clamp, Hub, Innertube, Rimtape, Spoke/nipple, Tire, Valve, Wheel, Battery, Charge port, Charger, Controller, Display, Display cable, Engine power cable, Software, Front hinge, Hook on handlebar post, Spring mechanism, Bottomplate downside, Gripstop, Throttle, Throttle cable, Brake light, Battery compartment, Battery cover, Body panel, End cap, Footrest, Front wheel cover, Handrail, Helmet hook, Neck cover, Panel, Side panel, Windshield, Brake disc, Alarm, Cable hall sensor, DC-DC converter, ECU, Flasher, Horn, Foot pegs, Kickstand spring, Midstand, Suspension, Swing arm, Combination switch, Mirror, Turn signal, Battery lock, Power lock, Battery clip, Engine, IOT Module, Speed sensor, Power button, Complete system, Chain wheel protector, Tensioner, Engine bracket, IOT CAN cable, USB charger, Carrier strip, Shimmy damper, SP mount, Protector
    - position: it needs to be one of these - front, rear, left, right, or null (when it's not applicable)
    - likely_service: it needs to be on eof these - Adjust, Repair, Replace, Grease, Lubricate, Tension, Tighten, Sticker, Pump, True, Add new, Bleed
    Respond in JSON format like:
    {{
        "bike_type": "Deluxe 7",
        "part_category": "Drivetrain",
        "part_name": "chain",
        "position": "null",
        "likely_service": "replace"
    }}

    Respond in this format:
    {parser.get_format_instructions()}
    """.strip()

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=last_msg)
    ])
    structured = parser.parse(result.content)

    incomplete = any(
        getattr(structured, field) is None
        for field in ["bike_type", "part_category", "part_name", "likely_service"]
    )
    current_agent = "speech_to_text" if incomplete else "repair_diagnostics"

    new_msg = AIMessage(content=str(structured.model_dump()))

    return {
        **state,
        "messages": state["messages"] + [new_msg],
        "current_agent": current_agent
    }


def route_from_issue_classifier(state: dict) -> str:
    return state["current_agent"]


def repair_diagnostics_node(state: dict) -> dict:
    print("TODO: implement repair diagnostics")
    return {
        **state,
        "current_agent": "final_response"
    }


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
    elif "repair" in content or "diagnose" in content:
        agent = "repair_diagnostics"
    # else:
    #     agent = "issue_classifier"

    return State(messages=state.messages, current_agent=agent)

# 5. Build LangGraph

graph = StateGraph(State)

# Add nodes
graph.add_node("speech_to_text", speech_to_text_node)
graph.add_node("issue_classifier", issue_classifier_node)
graph.add_node("repair_diagnostics", repair_diagnostics_node)

# Future nodes would go here:
# graph.add_node("clarification", clarification_node)
# graph.add_node("repair_vs_replace", repair_vs_replace_node)

# Add edges
graph.set_entry_point("speech_to_text")
graph.add_edge("speech_to_text", "issue_classifier")
graph.add_edge("issue_classifier", END)
graph.add_conditional_edges(
    "issue_classifier",
    route_from_issue_classifier,
    {
        "speech_to_text": "speech_to_text",           # loop back to speech input
        "repair_diagnostics": "repair_diagnostics"    # proceed to the next step
    }
)

# Terminate at placeholder for now
# graph.add_edge("speech_to_text", END)

app = graph.compile()
