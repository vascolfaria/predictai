import os
import sys
from typing import Literal, Optional
from dotenv import load_dotenv
from utils_functions.issue_classifier import IssueClassification
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from utils_functions.audio_tools import record_audio_to_wav

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()


# 1. State definition
class State(MessagesState):
    current_agent: Literal[
        "audio_intro_node",
        "audio_record_node",
        "audio_process_node",
        "issue_classifier",
        "repair_diagnostics"
    ]
    audio_path: Optional[str] = None
    transcription: Optional[str] = None


parser = PydanticOutputParser(pydantic_object=IssueClassification)

OpenAI.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4")
client = OpenAI()

sys_msg = SystemMessage(content="You are a helpful assistant trying to understand what is wrong with the member's bike. If the issue_classifier_node does not have enough information, ask the member for more feedback")


def audio_intro_node(state: dict) -> dict:
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content="🎙️ Alright, we are now recording for 10 seconds..")]
    }

def audio_record_node(state: dict) -> dict:
    print("[DEBUG] Running audio_record_node")
    audio_path = record_audio_to_wav(duration=3)
    print("[DEBUG] Audio path recorded:", audio_path)
    return {
        **state,
        "audio_path": audio_path,
        "messages": state["messages"] + [AIMessage(content="🔴 Recording finished, transcribing..")]
    }


def audio_process_node(state: dict) -> dict:
    print(state)
    audio_path = state.get("audio_path")
    if not audio_path:
        raise ValueError("Missing audio_path. Did you skip audio_record_node?")

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)

    transcription_text = transcript.text
    return {
        **state,
        "transcription": transcription_text,
        "messages": state["messages"] + [HumanMessage(content=transcription_text)]
    }

# Agent: Issue classifier
def issue_classifier_node(state: State) -> State:
    # Grab the last AI message (from audio_process_node)
    last_msg = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None
    )

    if not last_msg:
        raise ValueError("No message found for issue classification")


    prompt = f"""
    You are a technical classification assistant. Your job is to extract structured information from a customer's natural language description of a bike problem.
    You need to translate the text of the customer into categories. If you are not 100% sure of what it is, try to guess.
    
    If you do not know the value for a field, output null.

    Here are the fields you need to extract and the possible values of each one based on the text given by the member:
    - bike_type: Deluxe 7, Original 1, Original 1+, Power 1, Power 7 or Power Plus
    - part_category: Brakes, Drivetrain, Fenders, Frame, Gears, Handlebar, Light, Lock, Saddle, Wheel, Electrics, Folding, Body & Panel
    - part_name: Brake cable/hose, Brake lever, Brake pads, Brake unit/caliper, Bottom bracket, Chain, Chain wheel, Chainguard, Crank, Pedals, Sprocket, Fender, Fender stay, Barcode, Carrier, Carrier bracket, Carrier bumper, Frame/panel, Front fork, Kickstand, Kickstand foot, Cassette-joint, Gears, Shifter, Shifter cable, Bell, Grips, Handlebar, Headset, Headset cover, Stem, Bye bye battery, Cable, Light, Magnet, Reflector, Chain lock, Frame lock, Saddle, Saddle clamp, Seatpost, Seatpost clamp, Hub, Innertube, Rimtape, Spoke/nipple, Tire, Valve, Wheel, Battery, Charge port, Charger, Controller, Display, Display cable, Engine power cable, Software, Front hinge, Hook on handlebar post, Spring mechanism, Bottomplate downside, Gripstop, Throttle, Throttle cable, Brake light, Battery compartment, Battery cover, Body panel, End cap, Footrest, Front wheel cover, Handrail, Helmet hook, Neck cover, Panel, Side panel, Windshield, Brake disc, Alarm, Cable hall sensor, DC-DC converter, ECU, Flasher, Horn, Foot pegs, Kickstand spring, Midstand, Suspension, Swing arm, Combination switch, Mirror, Turn signal, Battery lock, Power lock, Battery clip, Engine, IOT Module, Speed sensor, Power button, Complete system, Chain wheel protector, Tensioner, Engine bracket, IOT CAN cable, USB charger, Carrier strip, Shimmy damper, SP mount, Protector
    - position: front, rear, left, right, or null (when it's not applicable)
    - likely_service: Adjust, Repair, Replace, Grease, Lubricate, Tension, Tighten, Sticker, Pump, True, Add new, Bleed
    
    Respond in JSON format, like this as an example:
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

    try:
        structured = parser.parse(result.content)
    except Exception:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(
                    content="Thanks! I need a bit more info to help — could you tell me which part of the bike this affects, and what kind of bike you have?")
            ],
            "current_agent": "wait_for_human"
        }

    # Check for incomplete fields
    incomplete = any(
        getattr(structured, field) is None
        for field in ["bike_type", "part_category", "part_name", "likely_service"]
    )

    if incomplete:
        followup_msg = AIMessage(
            content="Thanks! I need a bit more info to help — could you tell me which part of the bike this affects, and what kind of bike you have?"
        )
        return {
            **state,
            "messages": state["messages"] + [followup_msg],
            "current_agent": "audio_record_node"
        }

    # We're good to go
    new_msg = AIMessage(content=str(structured.model_dump()))
    return {
        **state,
        "messages": state["messages"] + [new_msg],
        "current_agent": "repair_diagnostics"
    }

def wait_for_human_node(state: State) -> State:
    return state


def needs_more_info(state: State) -> bool:
    last_ai_msg = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)),
        None
    )

    if not last_ai_msg:
        return True  # No AI response = not enough info

    try:
        structured = parser.parse(last_ai_msg.content)
    except Exception:
        return True  # Parsing failed = not valid info

    return any(
        getattr(structured, field) in (None, "null", "")
        for field in ["bike_type", "part_category", "part_name", "likely_service"]
    )


def repair_diagnostics_node(state: dict) -> dict:
    print("TODO: implement repair diagnostics")
    return {
        **state,
        "current_agent": "final_response"
    }


def route_from_issue_classifier(state: State) -> str:
    return "wait_for_human" if needs_more_info(state) else "repair_diagnostics"


# 5. Build LangGraph

graph = StateGraph(State)

# Add nodes
graph.add_node("audio_intro", audio_intro_node)
graph.add_node("audio_record", audio_record_node)
graph.add_node("audio_process", audio_process_node)
graph.add_node("issue_classifier", issue_classifier_node)
graph.add_node("repair_diagnostics", repair_diagnostics_node)
graph.add_node("wait_for_human", wait_for_human_node)

# Future nodes would go here:
# graph.add_node("clarification", clarification_node)
# graph.add_node("repair_vs_replace", repair_vs_replace_node)

# Add edges
graph.set_entry_point("audio_intro")
graph.add_edge("audio_intro", "audio_record")
graph.add_edge("audio_record", "audio_process")
graph.add_edge("audio_process", "issue_classifier")
graph.add_edge("wait_for_human", "audio_record")
graph.add_conditional_edges(
    "issue_classifier",
    route_from_issue_classifier,
    {
        "wait_for_human": "wait_for_human",
        "repair_diagnostics": "repair_diagnostics"
    }
)
graph.add_edge("repair_diagnostics", END)


# Terminate at placeholder for now
# graph.add_edge("repair_diagnostics", END)

app = graph.compile()
