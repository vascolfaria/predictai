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
from langgraph.checkpoint.memory import MemorySaver
from utils_functions.show_image import display_issues


from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()
memory = MemorySaver()


# 1. State definition
class State(MessagesState):
    current_agent: Literal[
        "audio_intro_node",
        "audio_record_node",
        "audio_process_node",
        "issue_classifier"
    ]
    audio_path: Optional[str] = None

    transcription: Optional[str] = None
    collected_info: Optional[dict] = None
    conversation_round: int = 0

parser = PydanticOutputParser(pydantic_object=IssueClassification)

OpenAI.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4")
client = OpenAI()

sys_msg = SystemMessage(content="You are a helpful assistant trying to understand what is wrong with the member's bike. If the issue_classifier_node does not have enough information, ask the member for more feedback")

def update_issues_node(state: State) -> State:
    return {
        **state,
        "messages": state["messages"] + [
            AIMessage(content="🔁 Okay, please tell me which parts need to be updated.")
        ]
    }



def render_issues_node(state: dict) -> dict:
    """Render issues with images based on collected information"""
    
    # Get the collected information from the state
    collected_info = state.get("collected_info", {})
    
    if not collected_info:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="❌ No issue information available to display.")
            ]
        }
    
    # Convert single issue to list format expected by display_issues
    issues = [collected_info] if isinstance(collected_info, dict) else collected_info
    
    try:
        # Generate markdown with images
        markdown_output = display_issues(issues)
        image_message = AIMessage(content=markdown_output)
        
        print(f"[DEBUG] Generated markdown output: {len(markdown_output)} characters")
        
        return {
            **state,
            "messages": state["messages"] + [image_message]
        }
    except Exception as e:
        print(f"[ERROR] Failed to render issues: {e}")
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=f"❌ Error displaying part information: {str(e)}")
            ]
        }

def confirm_issues_node(state: State) -> State:
    confirm_msg = AIMessage(content="✅ Does this look correct? (yes / no / partially)")
    return {
        **state,
        "messages": state["messages"] + [confirm_msg]
    }

def handle_confirmation_node(state: State) -> str:
    last_human = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "").lower()
    if "yes" in last_human:
        return "next_agent"
    elif "partial" in last_human:
        return "update_issues_node"
    elif "no" in last_human:
        return "clarification"
    return "confirm_issues"  # fallback loop


def audio_intro_node(state: dict) -> dict:
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content="🎙️ Alright, we are now recording for 10 seconds..")]
    }

def audio_record_node(state: dict) -> dict:
    print("[DEBUG] Running audio_record_node")
    audio_path = record_audio_to_wav(duration=10)
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

    def extract_content(msg) -> str:
        content = msg.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return " ".join(str(item) for item in content if item)
        elif isinstance(content, dict):
            return str(content.get('text', content.get('content', str(content))))
        else:
            return str(content)

    user_messages = [
        extract_content(msg) for msg in state["messages"]
        if isinstance(msg, HumanMessage)
    ]

    if not user_messages:
        raise ValueError("No user messages found for classification")

    user_messages = [msg for msg in user_messages if msg.strip()]

    combined_info = " ".join(user_messages)

    previous_info = state.get("collected_info", {})


    prompt = f"""
    You are a technical classification assistant. Your job is to extract structured information from a customer's natural language description of a bike problem.
    You need to translate the text of the customer into categories. If you are not 100% sure of what it is but confident enough, try to guess.
    
    Previous information provided by the user: {previous_info if previous_info else "There is no information provided yet"}
    The new information provided by the user: {combined_info}
    
    If you do not know the value for a field, output null.
    
    More than one problem can be reported by the member. The bike type will always be the same.

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

    The name of this JSON is reported_problems.
    
    The reported_problems can be more than one dict if more than one problem is being reported by the member.
    
    Respond in this format:
    {parser.get_format_instructions()}
    """.strip()

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=combined_info)
    ])

    try:
        structured = parser.parse(result.content)
        structured_dict = structured.model_dump()
    except Exception:
        # Incrementar contador de tentativas
        new_round = state.get("conversation_round", 0) + 1
        return {
            **state,
            "conversation_round": new_round,
            "messages": state["messages"] + [
                AIMessage(
                    content="Thanks! I need a bit more info to help — could you tell me which part of the bike this affects, and what kind of bike you have?")
            ],
            "current_agent": "audio_intro"
        }

    updated_info = {**previous_info, **structured_dict}

    # Check for incomplete fields
    required_fields = ["bike_type", "part_category", "part_name", "likely_service"]
    incomplete_fields = [
        field for field in required_fields
        if updated_info.get(field) in [None, "null", ""]
    ]

    if incomplete_fields:
        # Gerar pergunta específica baseada no que falta
        missing_info = ", ".join(incomplete_fields)
        followup_msg = AIMessage(
            content=f"Great! I still need some more information: {missing_info}. Can you give me any more details about it?"
        )
        new_round = state.get("conversation_round", 0) + 1
        return {
            **state,
            "collected_info": updated_info,
            "conversation_round": new_round,
            "messages": state["messages"] + [followup_msg],
            "current_agent": "audio_intro"
        }

    # Temos tudo que precisamos!
    success_msg = AIMessage(
        content=f"Perfect! I identified the problem: {updated_info}."
    )
    return {
        **state,
        "collected_info": updated_info,
        "messages": state["messages"] + [success_msg],
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


def route_from_issue_classifier(state: State) -> str:
    collected = state.get("collected_info", {})
    required_fields = ["bike_type", "part_category", "part_name", "likely_service"]

    has_all_info = all(
        collected.get(field) not in [None, "null", ""]
        for field in required_fields
    )

    return "render_issues" if has_all_info else "audio_intro"


# 5. Build LangGraph

graph = StateGraph(State)

# Add nodes
graph.add_node("audio_intro", audio_intro_node)
graph.add_node("audio_record", audio_record_node)
graph.add_node("audio_process", audio_process_node)
graph.add_node("issue_classifier", issue_classifier_node)
graph.add_node("wait_for_human", wait_for_human_node)
graph.add_node("render_issues", render_issues_node)
graph.add_node("confirm_issues", confirm_issues_node)
graph.add_node("update_issues_node", wait_for_human_node)
graph.add_node("clarification", wait_for_human_node)
graph.add_node("next_agent", wait_for_human_node)

# Add edges
graph.set_entry_point("audio_intro")
graph.add_edge("audio_intro", "audio_record")
graph.add_edge("audio_record", "audio_process")
graph.add_edge("audio_process", "issue_classifier")
graph.add_conditional_edges(
    "issue_classifier",
    route_from_issue_classifier,
    {
        "audio_intro": "audio_intro",
        "render_issues": "render_issues"
    }
)
graph.add_edge("render_issues", END)
# Route directly from confirm_issues using handle_confirmation_node
graph.add_conditional_edges("confirm_issues", handle_confirmation_node, {
    "next_agent": "next_agent",
    "clarification": "clarification",
    "update_issues_node": "update_issues_node",
})
graph.add_edge("update_issues_node", "render_issues")
graph.add_edge("next_agent", END)  # Add termination


# Terminate at placeholder for now

app = graph.compile(checkpointer=memory)
