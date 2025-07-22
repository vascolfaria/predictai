"""
Multi-Agent LangGraph with Speech-to-Text Node (Swapfiets)
"""
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

llm = ChatOpenAI(model="gpt-4")

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
graph.add_node("speech_to_text", speech_to_text_node)
graph.add_node("issue_classifier", issue_classifier_node)

# Future nodes would go here:
# graph.add_node("clarification", clarification_node)
# graph.add_node("repair_diagnostics", repair_diagnostics_node)
# graph.add_node("repair_vs_replace", repair_vs_replace_node)

# Add edges
graph.set_entry_point("speech_to_text")
graph.add_edge("speech_to_text", "issue_classifier")
graph.add_edge("issue_classifier", END)

# Terminate at placeholder for now
# graph.add_edge("speech_to_text", END)

app = graph.compile()
