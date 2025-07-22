import os
import sys
import json
from typing import Literal, Optional
from dotenv import load_dotenv
from utils_functions.issue_classifier import IssueClassification
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from utils_functions.audio_tools import record_audio_to_wav
from langgraph.checkpoint.memory import MemorySaver

import random

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()
memory = MemorySaver()


# Load repair menu data
def load_repair_menu():
    """Load repair menu JSON data"""
    try:
        with open('repair_menu.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: repair_menu.json not found. Repair assessment will not be available.")
        return []


def extract_possible_values(repair_menu_data):
    """Extract all possible values for each field from the repair menu (original case)"""
    unique_values = {
        'bike_type': set(),
        'part_category': set(),
        'part_name': set(),
        'position': set(),
        'likely_service': set()
    }

    for item in repair_menu_data:
        unique_values['bike_type'].add(item['bike_type'])
        unique_values['part_category'].add(item['part_category'])
        unique_values['part_name'].add(item['part_name'])
        unique_values['position'].add(item['position'])
        unique_values['likely_service'].add(item['likely_service'])

    return {key: sorted(list(values)) for key, values in unique_values.items()}


repair_menu = load_repair_menu()
POSSIBLE_VALUES = extract_possible_values(repair_menu) if repair_menu else {}


# 1. State definition
class State(MessagesState):
    current_agent: Literal[
        "audio_intro_node",
        "audio_record_node",
        "audio_process_node",
        "issue_classifier",
        "repair_assessment"
    ]
    audio_path: Optional[str] = None
    transcription: Optional[str] = None
    collected_info: Optional[dict] = None
    conversation_round: int = 0
    repair_assessment: Optional[dict] = None


parser = PydanticOutputParser(pydantic_object=IssueClassification)

OpenAI.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4")
client = OpenAI()

sys_msg = SystemMessage(
    content="You are a helpful assistant trying to understand what is wrong with the member's bike. If the issue_classifier_node does not have enough information, ask the member for more feedback")


def intro_prompt_node(state: dict) -> dict:
    """Generate a varied intro message to ask the user what's wrong with their bike."""
    prompts = [
        "🚲 Hey there! What's going on with your bike today?",
        "🛠️ Let’s get your bike back in shape. Can you tell me what seems to be the problem?",
        "🎤 I'm all ears! What's wrong with your bike?",
        "👋 Hi! Could you describe the issue you're having with your bike?",
        "🔧 Tell me what’s not working as it should — I’ll do my best to help.",
        "💬 Let’s start with what’s bugging you about the bike.",
        "📣 Go ahead and describe the issue — I’ll take it from there.",
        "❓What seems off with your bike? Share the details, big or small!",
        "🚴‍♂️ Something not right? Let me know what’s wrong with the bike.",
        "🧰 Okay, ready when you are — what’s going on with the bike?"
    ]
    intro_message = random.choice(prompts)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=intro_message)]
    }

def suggest_closest_value(invalid_value, valid_options):
    """Suggest the closest valid option for an invalid value"""
    invalid_lower = invalid_value.lower()

    # First try exact case-insensitive match
    for option in valid_options:
        if option.lower() == invalid_lower:
            return option

    # Then try partial matches
    matches = []
    for option in valid_options:
        if invalid_lower in option.lower() or option.lower() in invalid_lower:
            matches.append(option)

    return matches[0] if matches else None


def validate_and_correct_classification(structured_dict):
    """
    Validate and attempt to auto-correct classification values
    Returns (corrected_dict, was_corrected, errors)
    """
    corrected = structured_dict.copy()
    errors = []
    was_corrected = False

    for field, value in structured_dict.items():
        if field in POSSIBLE_VALUES and value not in POSSIBLE_VALUES[field]:
            # Try to find a close match
            suggestion = suggest_closest_value(value, POSSIBLE_VALUES[field])
            if suggestion:
                corrected[field] = suggestion
                was_corrected = True
                print(f"[DEBUG] Auto-corrected {field}: '{value}' -> '{suggestion}'")
            else:
                errors.append(f"Invalid {field}: '{value}'. Must be one of: {POSSIBLE_VALUES[field]}")

    return corrected, was_corrected, errors


def safe_lower(value):
    """Safely convert value to lowercase, handling None and NULL cases"""
    if not value or value == 'NULL':
        return value
    return value.lower() if isinstance(value, str) else str(value).lower()


def extract_issue_details_lowercase(collected_info):
    """Extract and convert issue details to lowercase for comparison"""
    return {
        'bike_type': safe_lower(collected_info.get('bike_type')),
        'part_category': safe_lower(collected_info.get('part_category')),
        'part_name': safe_lower(collected_info.get('part_name')),
        'position': safe_lower(collected_info.get('position')),
        'likely_service': safe_lower(collected_info.get('likely_service'))
    }
    """Normalize field values for comparison"""
    if value is None:
        return "NULL"
    if isinstance(value, str):
        if value.lower() in ["null", "none", ""]:
            return "NULL"
        return value.strip()
    return str(value)


def find_repair_option(bike_type, part_category, part_name, position, likely_service):
    """
    Find matching repair option in the repair menu
    Returns the repair option dict if found, None otherwise
    """
    # Simple lowercase conversion for comparison
    search_values = {
        'bike_type': safe_lower(bike_type),
        'part_category': safe_lower(part_category),
        'part_name': safe_lower(part_name),
        'position': safe_lower(position),
        'likely_service': safe_lower(likely_service)
    }

    for option in repair_menu:
        # Convert menu values to lowercase for comparison
        menu_values = {
            'bike_type': safe_lower(option.get('bike_type')),
            'part_category': safe_lower(option.get('part_category')),
            'part_name': safe_lower(option.get('part_name')),
            'position': safe_lower(option.get('position')),
            'likely_service': safe_lower(option.get('likely_service'))
        }

        if all(search_values[key] == menu_values[key] for key in search_values):
            return option

    return None


def repair_assessment_node(state: State) -> State:
    """
    Assess whether the identified issue can be repaired by a swapper or needs replacement
    """
    collected_info = state.get("collected_info", {})

    if not collected_info:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="❌ No issue information found for repair assessment.")
            ]
        }

    # Extract issue details
    bike_type = collected_info.get('bike_type')
    part_category = collected_info.get('part_category')
    part_name = collected_info.get('part_name')
    position = collected_info.get('position')
    likely_service = collected_info.get('likely_service')

    # Find matching repair option
    repair_option = find_repair_option(bike_type, part_category, part_name, position, likely_service)

    if repair_option is None:
        assessment_msg = AIMessage(
            content=f"⚠️ **Repair Assessment**\n\n"
                    f"I couldn't find this specific repair combination in our service menu:\n"
                    f"• **Bike**: {bike_type}\n"
                    f"• **Part**: {part_name} ({part_category})\n"
                    f"• **Position**: {position if position and position != 'NULL' else 'Not applicable'}\n"
                    f"• **Service**: {likely_service}\n\n"
                    f"Please contact our support team for a manual assessment."
        )

        assessment_result = {
            "found_in_menu": False,
            "can_swapper_repair": None,
            "issue_details": collected_info,
            "message": "Not found in repair menu"
        }
    else:
        can_repair = repair_option.get('can_swapper_repair', 0) == 1

        if can_repair:
            status_emoji = "✅"
            status_text = "**Good news!** This can be repaired by one of our bike swappers."
            repair_info = "A swapper will be able to fix this issue for you without needing to replace your bike."
        else:
            status_emoji = "🔄"
            status_text = "This issue requires a **bike replacement**."
            repair_info = "Our swappers cannot repair this particular issue, so we'll arrange a replacement bike for you."

        position_text = position if position and position != 'NULL' else 'Not applicable'

        assessment_msg = AIMessage(
            content=f"{status_emoji} **Repair Assessment**\n\n"
                    f"{status_text}\n\n"
                    f"**Issue Details:**\n"
                    f"• **Bike Model**: {bike_type}\n"
                    f"• **Component**: {part_name} ({part_category})\n"
                    f"• **Location**: {position_text}\n"
                    f"• **Required Service**: {likely_service}\n\n"
                    f"**Next Steps:**\n"
                    f"{repair_info}\n\n"
                    f"Would you like me to proceed with booking this service?"
        )

        assessment_result = {
            "found_in_menu": True,
            "can_swapper_repair": can_repair,
            "issue_details": collected_info,
            "repair_option": repair_option,
            "message": "Repair assessment completed"
        }

    return {
        **state,
        "repair_assessment": assessment_result,
        "messages": state["messages"] + [assessment_msg]
    }



def confirm_issues_node(state: State) -> State:
    confirm_msg = AIMessage(content="✅ Does this look correct? (yes / no)")
    return {
        **state,
        "messages": state["messages"] + [confirm_msg]
    }


def handle_confirmation_node(state: State) -> str:
    last_human = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "").lower()
    if "yes" in last_human:
        return "repair_assessment"  # if positive
    elif "clarification" in last_human:
        return "clarification_node" #if negative
    return "success_node"  # success forever


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

    # Generate dynamic prompt with exact possible values from repair menu
    bike_types = ", ".join(POSSIBLE_VALUES.get('bike_type', []))
    part_categories = ", ".join(POSSIBLE_VALUES.get('part_category', []))
    part_names = ", ".join(POSSIBLE_VALUES.get('part_name', []))
    positions = ", ".join(POSSIBLE_VALUES.get('position', []))
    likely_services = ", ".join(POSSIBLE_VALUES.get('likely_service', []))

    prompt = f"""
    You are a technical classification assistant. Your job is to extract structured information from a customer's natural language description of a bike problem.
    You need to translate the text of the customer into categories. You MUST only use the exact values listed below - no variations or alternatives allowed.

    Previous information provided by the user: {previous_info if previous_info else "There is no information provided yet"}
    The new information provided by the user: {combined_info}

    IMPORTANT: You must only use the EXACT values listed below. Use exact capitalization as shown. If you cannot determine a field with confidence, use "NULL".

    EXACT POSSIBLE VALUES (use these ONLY):

    bike_type (choose exactly one): {bike_types}

    part_category (choose exactly one): {part_categories}

    part_name (choose exactly one): {part_names}

    position (choose exactly one): {positions}

    likely_service (choose exactly one): {likely_services}

    RULES:
    1. Use EXACT spelling and capitalization as shown above
    2. If unsure about a value, use "NULL" rather than guessing incorrectly
    3. Position should be "NULL" when location is not applicable to the part
    4. Choose the most specific part_name that matches the user's description

    Example response format:
    {{
        "bike_type": "Deluxe 7",
        "part_category": "Drivetrain", 
        "part_name": "Chain",
        "position": "NULL",
        "likely_service": "Replace"
    }}

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

        # Validate and attempt to auto-correct values
        corrected_dict, was_corrected, validation_errors = validate_and_correct_classification(structured_dict)

        if validation_errors:
            print(f"[DEBUG] Validation errors after correction: {validation_errors}")
            raise ValueError(f"Invalid classification values: {validation_errors}")

        if was_corrected:
            print(f"[DEBUG] Auto-corrected classification values")

        # Use the corrected dictionary
        structured_dict = corrected_dict

    except Exception as e:
        print(f"[DEBUG] Classification parsing/validation failed: {str(e)}")
        # Increment attempt counter
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
        # Generate specific question based on what's missing
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

    # We have all the information we need! Create an engaging summary
    bike_type = updated_info.get('bike_type', 'Unknown')
    part_name = updated_info.get('part_name', 'Unknown part')
    part_category = updated_info.get('part_category', '')
    position = updated_info.get('position', '')
    likely_service = updated_info.get('likely_service', 'service')

    # Format position nicely
    position_text = ""
    if position and position.lower() not in ['null', 'none', '']:
        position_text = f" {position.lower()}"

    # Format part description
    part_description = f"{position_text} {part_name.lower()}" if position_text else part_name.lower()

    # Make service description more natural
    service_descriptions = {
        'replace': 'needs to be replaced',
        'repair': 'needs to be repaired',
        'adjust': 'needs adjustment',
        'lubricate': 'needs lubrication',
        'tighten': 'needs to be tightened',
        'grease': 'needs greasing',
        'pump': 'needs to be pumped up',
        'tension': 'needs tension adjustment'
    }

    service_text = service_descriptions.get(likely_service.lower(), f"needs {likely_service.lower()}")

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

def clarification_node(state: dict) -> dict:
    """Ask the user to clarify the issue and loop back to the audio prompt."""
    prompts = [
        "🤔 Hmm, I’m not quite sure I understood that. Could you tell me again what's wrong with your bike?",
        "🧐 Could you clarify the issue a bit more? I'm ready to listen again.",
        "🔁 I didn’t catch enough to understand the problem. Let’s try once more — what's going on with your bike?",
        "🙋‍♀️ Mind repeating that for me? I need a bit more detail to help you out.",
        "🎧 I want to make sure I get it right — can you describe the issue again?"
    ]
    clarification_msg = random.choice(prompts)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=clarification_msg)]
    }

def success_node(state: dict) -> dict:
    """Say goodbye after the repair assessment is done."""
    farewells = [
        "✅ All set! Thanks for letting us know — we’ll take it from here 🚲",
        "👍 Got it! We’ll get on this right away. Have a great ride!",
        "👋 That’s everything I need for now. Thanks and take care!",
        "🚴 Your request is confirmed — enjoy your ride!",
        "🛠️ We’ve logged your issue. Thanks for your patience and trust!",
        "✨ Thanks! We’ll make sure your bike is in tip-top shape.",
        "🔧 All done! A swapper will be on it soon.",
        "📦 Thanks for the details. We’ll take it from here.",
        "🥳 That’s it! You’re all set.",
        "👊 Nice work — we’ve got everything we need now."
    ]
    goodbye_message = random.choice(farewells)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=goodbye_message)]
    }

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
graph.add_node("intro_prompt_node", intro_prompt_node)
graph.add_node("audio_intro", audio_intro_node)
graph.add_node("audio_record", audio_record_node)
graph.add_node("audio_process", audio_process_node)
graph.add_node("issue_classifier", issue_classifier_node)
graph.add_node("wait_for_human", wait_for_human_node)
graph.add_node("confirm_issues", confirm_issues_node)
graph.add_node("clarification_node", clarification_node)
graph.add_node("repair_assessment", repair_assessment_node)
graph.add_node("success_node", success_node)

# Add edges
graph.set_entry_point("intro_prompt_node")
graph.add_edge("intro_prompt_node", "audio_intro")
graph.add_edge("audio_intro", "audio_record")
graph.add_edge("audio_record", "audio_process")
graph.add_edge("audio_process", "issue_classifier")
graph.add_conditional_edges(
    "issue_classifier",
    route_from_issue_classifier,
    {
        "audio_intro": "audio_intro",
        "confirm_issues": "confirm_issues"
    }
)

# Route from confirm_issues
graph.add_conditional_edges("confirm_issues", handle_confirmation_node, {
    "repair_assessment": "repair_assessment",  # positive
    "clarification_node": "clarification_node"    # negative
})

graph.add_edge("clarification_node", "audio_intro")

graph.add_edge("repair_assessment", "success_node")
graph.add_edge("success_node", END)  # End after success

# Terminate at placeholder for now
app = graph.compile(checkpointer=memory)