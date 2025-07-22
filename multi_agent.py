import os
import sys
import json
from typing import Literal, Optional, List, Dict
from dotenv import load_dotenv
from utils_functions.issue_classifier import IssueClassification
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from utils_functions.audio_tools import record_audio_to_wav
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
import random

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()
memory = MemorySaver()


# Enhanced Pydantic model for multiple issues
class MultipleIssuesClassification(BaseModel):
    bike_type: str
    issues: List[IssueClassification]
    total_count: int


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


# State definition - back to simpler structure
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
    collected_info: Optional[dict] = None  # Back to single dict but with multiple issues
    conversation_round: int = 0
    repair_assessment: Optional[dict] = None


# Parsers
parser = PydanticOutputParser(pydantic_object=IssueClassification)
multi_parser = PydanticOutputParser(pydantic_object=MultipleIssuesClassification)

OpenAI.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4")
client = OpenAI()

sys_msg = SystemMessage(
    content="You are a helpful assistant trying to understand what is wrong with the member's bike. "
            "Listen carefully as they may describe multiple issues at once, and extract all problems mentioned.")


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


def generate_specific_feedback(user_input, previous_info, round_number):
    """
    Generate specific feedback based on what we could understand from user input
    """
    user_lower = user_input.lower()

    # Try to detect what they might be talking about
    detected_parts = []
    if any(word in user_lower for word in ['chain', 'chains']):
        detected_parts.append('chain')
    if any(word in user_lower for word in ['brake', 'brakes', 'braking']):
        detected_parts.append('brakes')
    if any(word in user_lower for word in ['light', 'lights', 'lamp']):
        detected_parts.append('lights')
    if any(word in user_lower for word in ['wheel', 'tire', 'rim']):
        detected_parts.append('wheel')
    if any(word in user_lower for word in ['gear', 'gears', 'shifting']):
        detected_parts.append('gears')
    if any(word in user_lower for word in ['pedal', 'pedals']):
        detected_parts.append('pedals')

    # Detect bike types mentioned
    detected_bikes = []
    for bike_type in POSSIBLE_VALUES.get('bike_type', []):
        if bike_type.lower() in user_lower:
            detected_bikes.append(bike_type)

    if round_number == 1:
        if detected_parts and detected_bikes:
            return (f"I can hear you mentioned your **{detected_bikes[0]}** has issues with the "
                    f"**{', '.join(detected_parts)}**. Could you be more specific about what exactly is wrong? "
                    f"For example: 'the chain is loose' or 'the front brake squeaks'?")
        elif detected_parts:
            return (f"I understand there's something wrong with the **{', '.join(detected_parts)}**. "
                    f"What type of bike do you have, and what exactly is the problem? "
                    f"Try saying something like: 'My Deluxe 7 chain is broken'")
        elif detected_bikes:
            return (f"Got it, you have a **{detected_bikes[0]}**! What specific parts are having problems? "
                    f"Tell me which components need attention and what's wrong with them.")
        else:
            return ("I want to help but I'm having trouble understanding the specific issues. "
                    "Could you tell me: What type of bike you have and which part is broken? "
                    "For example: 'My Deluxe 7 front brake doesn't work'")
    else:
        return ("Let me try again - could you clearly state: your bike model and what's broken? "
                "Example: 'Classic 3 rear wheel is wobbly and chain skips'")


def generate_bike_type_feedback(user_input, previous_info, round_number):
    """
    Generate feedback when bike type is missing or unclear
    """
    user_lower = user_input.lower()

    # Check if they mentioned any bike-related words but not the specific type
    has_bike_context = any(word in user_lower for word in ['bike', 'bicycle', 'cycle'])

    available_bikes = POSSIBLE_VALUES.get('bike_type', [])
    bike_examples = ', '.join(available_bikes[:3]) if available_bikes else "Deluxe 7, Classic 3"

    if has_bike_context:
        return (f"I can see you're describing bike problems, but I need to know which bike model you have. "
                f"Is it a **{bike_examples}**, or another type? "
                f"Please say something like: 'I have a {available_bikes[0] if available_bikes else 'Deluxe 7'} and...'")
    else:
        return (f"I need to know your bike model first. Do you have a **{bike_examples}** or a different type? "
                f"Once I know the model, tell me what's wrong with it.")


def generate_detailed_issue_feedback(bike_type, complete_issues, incomplete_issues, round_number):
    """
    Generate detailed feedback acknowledging what we know and asking for what's missing
    Can handle multiple incomplete issues simultaneously
    """
    feedback_parts = []

    # Acknowledge the bike type
    feedback_parts.append(f"Perfect! I've got your **{bike_type}** identified.")

    # Acknowledge complete issues
    if complete_issues:
        if len(complete_issues) == 1:
            issue = complete_issues[0]['issue']
            feedback_parts.append(
                f"✅ **Issue {complete_issues[0]['number']}:** {issue.get('part_name')} problem - got it!")
        else:
            feedback_parts.append(f"✅ I've identified **{len(complete_issues)} issues** clearly.")

    # Handle multiple incomplete issues simultaneously
    if len(incomplete_issues) == 1:
        # Single incomplete issue - detailed approach
        issue_info = incomplete_issues[0]
        issue = issue_info['issue']
        missing = issue_info['missing']

        # Get what we do know about this issue
        known_parts = []
        if issue.get('part_name') and issue.get('part_name') != 'NULL':
            known_parts.append(f"it's the {issue.get('part_name')}")
        if issue.get('part_category') and issue.get('part_category') != 'NULL':
            known_parts.append(f"it's a {issue.get('part_category')} component")

        if known_parts:
            feedback_parts.append(f"For the other issue, I know {' and '.join(known_parts)}.")

        # Ask specifically for what's missing
        if 'likely_service' in missing:
            feedback_parts.append("What exactly is wrong with it? Is it broken, loose, not working, making noise?")
        elif 'part_name' in missing:
            feedback_parts.append("Which specific part has the problem? Like 'chain', 'brake', 'light', etc.?")
        elif 'part_category' in missing:
            feedback_parts.append("What type of component is having issues?")

    elif len(incomplete_issues) <= 3:
        # Multiple incomplete issues - ask about them simultaneously
        feedback_parts.append(f"I need more details about **{len(incomplete_issues)} other issues** you mentioned.")

        # Create specific questions for each incomplete issue
        issue_questions = []
        for i, issue_info in enumerate(incomplete_issues):
            issue_num = issue_info['number']
            issue = issue_info['issue']
            missing = issue_info['missing']

            # Build what we know about this issue
            known_info = []
            if issue.get('part_name') and issue.get('part_name') != 'NULL':
                known_info.append(issue.get('part_name'))
            if issue.get('part_category') and issue.get('part_category') != 'NULL':
                known_info.append(f"({issue.get('part_category')})")

            # Build the question based on what's missing
            if 'part_name' in missing and 'likely_service' in missing:
                if known_info:
                    question = f"**Issue {issue_num}** ({' '.join(known_info)}): which part and what's wrong?"
                else:
                    question = f"**Issue {issue_num}**: which part has a problem and what's wrong with it?"
            elif 'part_name' in missing:
                question = f"**Issue {issue_num}**: which specific part has the problem?"
            elif 'likely_service' in missing:
                if known_info:
                    question = f"**Issue {issue_num}** ({' '.join(known_info)}): what exactly is wrong?"
                else:
                    question = f"**Issue {issue_num}**: what exactly is the problem?"
            elif 'part_category' in missing:
                question = f"**Issue {issue_num}**: what type of component is this?"
            else:
                question = f"**Issue {issue_num}**: need more details"

            issue_questions.append(question)

        # Combine all questions
        if len(issue_questions) == 2:
            feedback_parts.append(f"{issue_questions[0]} And {issue_questions[1].lower()}")
        else:
            feedback_parts.append(f"{'; '.join(issue_questions)}.")

        # Add helpful example
        feedback_parts.append("For example: 'Issue 1 is front brake squeaking, Issue 2 is chain loose'")

    else:
        # Too many incomplete issues - simplify the approach
        feedback_parts.append(f"I detected **{len(incomplete_issues)} issues** but need clearer details.")
        feedback_parts.append("Could you list them clearly? For example:")
        feedback_parts.append(
            "'First issue: front brake squeaks. Second issue: chain is loose. Third issue: rear light broken.'")

    return " ".join(feedback_parts)


def generate_specific_feedback(user_input, previous_info, round_number):
    """
    Generate specific feedback based on what we could understand from user input
    """
    user_lower = user_input.lower()

    # Try to detect what they might be talking about
    detected_parts = []
    if any(word in user_lower for word in ['chain', 'chains']):
        detected_parts.append('chain')
    if any(word in user_lower for word in ['brake', 'brakes', 'braking']):
        detected_parts.append('brakes')
    if any(word in user_lower for word in ['light', 'lights', 'lamp']):
        detected_parts.append('lights')
    if any(word in user_lower for word in ['wheel', 'tire', 'rim']):
        detected_parts.append('wheel')
    if any(word in user_lower for word in ['gear', 'gears', 'shifting']):
        detected_parts.append('gears')
    if any(word in user_lower for word in ['pedal', 'pedals']):
        detected_parts.append('pedals')

    # Detect bike types mentioned
    detected_bikes = []
    for bike_type in POSSIBLE_VALUES.get('bike_type', []):
        if bike_type.lower() in user_lower:
            detected_bikes.append(bike_type)

    if round_number == 1:
        if detected_parts and detected_bikes:
            return (f"I can hear you mentioned your **{detected_bikes[0]}** has issues with the "
                    f"**{', '.join(detected_parts)}**. Could you be more specific about what exactly is wrong? "
                    f"For example: 'the chain is loose' or 'the front brake squeaks'?")
        elif detected_parts:
            return (f"I understand there's something wrong with the **{', '.join(detected_parts)}**. "
                    f"What type of bike do you have, and what exactly is the problem? "
                    f"Try saying something like: 'My Deluxe 7 chain is broken'")
        elif detected_bikes:
            return (f"Got it, you have a **{detected_bikes[0]}**! What specific parts are having problems? "
                    f"Tell me which components need attention and what's wrong with them.")
        else:
            return ("I want to help but I'm having trouble understanding the specific issues. "
                    "Could you tell me: What type of bike you have and which part is broken? "
                    "For example: 'My Deluxe 7 front brake doesn't work'")
    else:
        return ("Let me try again - could you clearly state: your bike model and what's broken? "
                "Example: 'Classic 3 rear wheel is wobbly and chain skips'")


def generate_bike_type_feedback(user_input, previous_info, round_number):
    """
    Generate feedback when bike type is missing or unclear
    """
    user_lower = user_input.lower()

    # Check if they mentioned any bike-related words but not the specific type
    has_bike_context = any(word in user_lower for word in ['bike', 'bicycle', 'cycle'])

    available_bikes = POSSIBLE_VALUES.get('bike_type', [])
    bike_examples = ', '.join(available_bikes[:3]) if available_bikes else "Deluxe 7, Classic 3"

    if has_bike_context:
        return (f"I can see you're describing bike problems, but I need to know which bike model you have. "
                f"Is it a **{bike_examples}**, or another type? "
                f"Please say something like: 'I have a {available_bikes[0] if available_bikes else 'Deluxe 7'} and...'")
    else:
        return (f"I need to know your bike model first. Do you have a **{bike_examples}** or a different type? "
                f"Once I know the model, tell me what's wrong with it.")


def generate_detailed_issue_feedback(bike_type, complete_issues, incomplete_issues, round_number):
    """
    Generate detailed feedback acknowledging what we know and asking for what's missing
    Can handle multiple incomplete issues simultaneously
    """
    feedback_parts = []

    # Acknowledge the bike type
    feedback_parts.append(f"Perfect! I've got your **{bike_type}** identified.")

    # Acknowledge complete issues
    if complete_issues:
        if len(complete_issues) == 1:
            issue = complete_issues[0]['issue']
            feedback_parts.append(
                f"✅ **Issue {complete_issues[0]['number']}:** {issue.get('part_name')} problem - got it!")
        else:
            feedback_parts.append(f"✅ I've identified **{len(complete_issues)} issues** clearly.")

    # Handle multiple incomplete issues simultaneously
    if len(incomplete_issues) == 1:
        # Single incomplete issue - detailed approach
        issue_info = incomplete_issues[0]
        issue = issue_info['issue']
        missing = issue_info['missing']

        # Get what we do know about this issue
        known_parts = []
        if issue.get('part_name') and issue.get('part_name') != 'NULL':
            known_parts.append(f"it's the {issue.get('part_name')}")
        if issue.get('part_category') and issue.get('part_category') != 'NULL':
            known_parts.append(f"it's a {issue.get('part_category')} component")

        if known_parts:
            feedback_parts.append(f"For the other issue, I know {' and '.join(known_parts)}.")

        # Ask specifically for what's missing
        if 'likely_service' in missing:
            feedback_parts.append("What exactly is wrong with it? Is it broken, loose, not working, making noise?")
        elif 'part_name' in missing:
            feedback_parts.append("Which specific part has the problem? Like 'chain', 'brake', 'light', etc.?")
        elif 'part_category' in missing:
            feedback_parts.append("What type of component is having issues?")

    elif len(incomplete_issues) <= 3:
        # Multiple incomplete issues - ask about them simultaneously
        feedback_parts.append(f"I need more details about **{len(incomplete_issues)} other issues** you mentioned.")

        # Create specific questions for each incomplete issue
        issue_questions = []
        for i, issue_info in enumerate(incomplete_issues):
            issue_num = issue_info['number']
            issue = issue_info['issue']
            missing = issue_info['missing']

            # Build what we know about this issue
            known_info = []
            if issue.get('part_name') and issue.get('part_name') != 'NULL':
                known_info.append(issue.get('part_name'))
            if issue.get('part_category') and issue.get('part_category') != 'NULL':
                known_info.append(f"({issue.get('part_category')})")

            # Build the question based on what's missing
            if 'part_name' in missing and 'likely_service' in missing:
                if known_info:
                    question = f"**Issue {issue_num}** ({' '.join(known_info)}): which part and what's wrong?"
                else:
                    question = f"**Issue {issue_num}**: which part has a problem and what's wrong with it?"
            elif 'part_name' in missing:
                question = f"**Issue {issue_num}**: which specific part has the problem?"
            elif 'likely_service' in missing:
                if known_info:
                    question = f"**Issue {issue_num}** ({' '.join(known_info)}): what exactly is wrong?"
                else:
                    question = f"**Issue {issue_num}**: what exactly is the problem?"
            elif 'part_category' in missing:
                question = f"**Issue {issue_num}**: what type of component is this?"
            else:
                question = f"**Issue {issue_num}**: need more details"

            issue_questions.append(question)

        # Combine all questions
        if len(issue_questions) == 2:
            feedback_parts.append(f"{issue_questions[0]} And {issue_questions[1].lower()}")
        else:
            feedback_parts.append(f"{'; '.join(issue_questions)}.")

        # Add helpful example
        feedback_parts.append("For example: 'Issue 1 is front brake squeaking, Issue 2 is chain loose'")

    else:
        # Too many incomplete issues - simplify the approach
        feedback_parts.append(f"I detected **{len(incomplete_issues)} issues** but need clearer details.")
        feedback_parts.append("Could you list them clearly? For example:")
        feedback_parts.append(
            "'First issue: front brake squeaks. Second issue: chain is loose. Third issue: rear light broken.'")

    return " ".join(feedback_parts)


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
    Assess multiple issues and determine if bike needs replacement or can be repaired
    """
    collected_info = state.get("collected_info", {})

    if not collected_info:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="❌ No issue information found for repair assessment.")
            ]
        }

    bike_type = collected_info.get('bike_type')
    issues = collected_info.get('issues', [])

    if not issues:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="❌ No issues found in the collected information.")
            ]
        }

    # Assess each issue
    assessments = []
    repairable_issues = []
    replacement_issues = []
    unknown_issues = []

    for i, issue in enumerate(issues):
        # Each issue should already have bike_type, but ensure it's set
        issue_data = {**issue, 'bike_type': bike_type}

        part_category = issue_data.get('part_category')
        part_name = issue_data.get('part_name')
        position = issue_data.get('position')
        likely_service = issue_data.get('likely_service')

        # Find matching repair option
        repair_option = find_repair_option(bike_type, part_category, part_name, position, likely_service)

        if repair_option is None:
            assessment = {
                "issue_number": i + 1,
                "found_in_menu": False,
                "can_swapper_repair": None,
                "issue_details": issue_data,
                "message": "Not found in repair menu"
            }
            unknown_issues.append(assessment)
        else:
            can_repair = repair_option.get('can_swapper_repair', 0) == 1
            assessment = {
                "issue_number": i + 1,
                "found_in_menu": True,
                "can_swapper_repair": can_repair,
                "issue_details": issue_data,
                "repair_option": repair_option,
                "message": "Assessment completed"
            }

            if can_repair:
                repairable_issues.append(assessment)
            else:
                replacement_issues.append(assessment)

        assessments.append(assessment)

    # Determine overall outcome
    total_issues = len(issues)
    needs_replacement = len(replacement_issues) > 0

    # Create detailed message
    if needs_replacement:
        outcome_emoji = "🔄"
        outcome_title = "**Bike Replacement Required**"
        outcome_explanation = ("Since at least one issue cannot be repaired by our swappers, "
                               "we'll arrange a **replacement bike** for you.")
    else:
        outcome_emoji = "✅"
        outcome_title = "**Bike Can Be Repaired**"
        outcome_explanation = "All issues can be fixed by our bike swappers!"

    # Build detailed issue breakdown
    issue_details = []
    for i, issue in enumerate(issues):
        position_text = issue.get('position', 'NULL')
        if position_text == 'NULL':
            position_text = 'Not applicable'

        issue_details.append(
            f"**Issue {i + 1}:** {issue.get('part_name')} ({issue.get('part_category')}) - "
            f"{issue.get('likely_service')} - Location: {position_text}"
        )

    # Summary statistics
    summary_stats = []
    if repairable_issues:
        summary_stats.append(
            f"• {len(repairable_issues)} issue{'s' if len(repairable_issues) != 1 else ''} **repairable by swapper**")
    if replacement_issues:
        summary_stats.append(
            f"• {len(replacement_issues)} issue{'s' if len(replacement_issues) != 1 else ''} **require replacement**")
    if unknown_issues:
        summary_stats.append(
            f"• {len(unknown_issues)} issue{'s' if len(unknown_issues) != 1 else ''} **need manual assessment**")

    assessment_msg = AIMessage(
        content=f"{outcome_emoji} **Assessment Complete - {bike_type}**\n\n"
                f"{outcome_title}\n\n"
                f"**Issues Identified ({total_issues} total):**\n" +
                "\n".join(issue_details) + "\n\n" +
                f"**Summary:**\n" + "\n".join(summary_stats) + "\n\n" +
                f"**Decision:** {outcome_explanation}\n\n"
                f"Would you like me to proceed with booking this service?"
    )

    assessment_result = {
        "bike_type": bike_type,
        "total_issues": total_issues,
        "needs_replacement": needs_replacement,
        "repairable_count": len(repairable_issues),
        "replacement_count": len(replacement_issues),
        "unknown_count": len(unknown_issues),
        "detailed_assessments": assessments,
        "issues": issues
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
        "messages": state["messages"] + [AIMessage(
            content="🎙️ Please describe all the issues you're experiencing with your bike. Recording for 10 seconds...")]
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


def issue_classifier_node(state: State) -> State:
    """
    Enhanced classifier that can identify multiple issues from a single input
    """

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

    # Generate dynamic prompt for multi-issue classification
    bike_types = ", ".join(POSSIBLE_VALUES.get('bike_type', []))
    part_categories = ", ".join(POSSIBLE_VALUES.get('part_category', []))
    part_names = ", ".join(POSSIBLE_VALUES.get('part_name', []))
    positions = ", ".join(POSSIBLE_VALUES.get('position', []))
    likely_services = ", ".join(POSSIBLE_VALUES.get('likely_service', []))

    prompt = f"""
    You are a technical classification assistant. Extract structured information from a customer's description of bike problems.
    The user may describe MULTIPLE issues in their message - identify and classify each one separately.

    User's description: {combined_info}
    Previous information: {previous_info if previous_info else "None"}

    IMPORTANT: 
    1. Identify ALL issues mentioned (brake problems, chain issues, light problems, etc.)
    2. The bike_type should be the SAME for all issues (one bike, multiple problems)
    3. Create separate issue objects for each distinct problem
    4. Use EXACT values from the lists below - no variations allowed
    5. If you cannot determine a field with confidence, use "NULL"

    EXACT POSSIBLE VALUES:
    bike_type: {bike_types}
    part_category: {part_categories}
    part_name: {part_names}
    position: {positions}
    likely_service: {likely_services}

    Expected JSON format:
    {{
        "bike_type": "Deluxe 7",
        "issues": [
            {{
                "part_category": "Drivetrain",
                "part_name": "Chain", 
                "position": "NULL",
                "likely_service": "Replace"
            }},
            {{
                "part_category": "Electrical",
                "part_name": "Front Light",
                "position": "Front",
                "likely_service": "Replace"
            }}
        ],
        "total_count": 2
    }}

    If only one issue is mentioned, still use the issues array format with one item.
    """.strip()

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=combined_info)
    ])

    try:
        # Parse the JSON response directly
        import re
        json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            structured_dict = json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")

        # Validate bike_type
        bike_type = structured_dict.get('bike_type')
        if bike_type not in POSSIBLE_VALUES.get('bike_type', []):
            suggestion = suggest_closest_value(bike_type, POSSIBLE_VALUES.get('bike_type', []))
            if suggestion:
                structured_dict['bike_type'] = suggestion
                print(f"[DEBUG] Auto-corrected bike_type: '{bike_type}' -> '{suggestion}'")

        # Validate each issue
        validated_issues = []
        for issue in structured_dict.get('issues', []):
            corrected_issue, was_corrected, validation_errors = validate_and_correct_classification(issue)
            if validation_errors:
                print(f"[DEBUG] Issue validation errors: {validation_errors}")
                continue  # Skip invalid issues but keep valid ones
            validated_issues.append(corrected_issue)

        if not validated_issues:
            raise ValueError("No valid issues found after validation")

        structured_dict['issues'] = validated_issues
        structured_dict['total_count'] = len(validated_issues)

    except Exception as e:
        print(f"[DEBUG] Multi-issue classification failed: {str(e)}")
        # Generate specific feedback based on what we could understand
        new_round = state.get("conversation_round", 0) + 1

        # Try to extract any partial information from the user's message
        feedback_msg = generate_specific_feedback(combined_info, previous_info, new_round)

        return {
            **state,
            "conversation_round": new_round,
            "messages": state["messages"] + [AIMessage(content=feedback_msg)],
            "current_agent": "audio_intro"
        }

    # Check if we have enough information
    bike_type = structured_dict.get('bike_type')
    issues = structured_dict.get('issues', [])

    if not bike_type or bike_type == 'NULL' or not issues:
        new_round = state.get("conversation_round", 0) + 1
        feedback_msg = generate_bike_type_feedback(combined_info, previous_info, new_round)
        return {
            **state,
            "conversation_round": new_round,
            "messages": state["messages"] + [AIMessage(content=feedback_msg)],
            "current_agent": "audio_intro"
        }

    # Check if individual issues are complete and generate specific feedback
    incomplete_issues = []
    complete_issues = []

    for i, issue in enumerate(issues):
        required_fields = ["part_category", "part_name", "likely_service"]
        missing = [field for field in required_fields if issue.get(field) in [None, "null", "", "NULL"]]

        if missing:
            incomplete_issues.append({
                "number": i + 1,
                "issue": issue,
                "missing": missing
            })
        else:
            complete_issues.append({
                "number": i + 1,
                "issue": issue
            })

    if incomplete_issues:
        new_round = state.get("conversation_round", 0) + 1
        feedback_msg = generate_detailed_issue_feedback(
            bike_type, complete_issues, incomplete_issues, new_round
        )
        return {
            **state,
            "conversation_round": new_round,
            "messages": state["messages"] + [AIMessage(content=feedback_msg)],
            "current_agent": "audio_intro"
        }

    # Success! Create summary message
    issue_count = len(issues)
    issue_summaries = []
    for i, issue in enumerate(issues):
        part_name = issue.get('part_name', 'Unknown part')
        part_category = issue.get('part_category', '')
        likely_service = issue.get('likely_service', 'service')
        issue_summaries.append(f"**Issue {i + 1}:** {part_name} ({part_category}) needs {likely_service.lower()}")

    success_msg = AIMessage(
        content=f"🔧 **Perfect!** I've identified **{issue_count} issue{'s' if issue_count != 1 else ''}** with your **{bike_type}**:\n\n" +
                "\n".join(issue_summaries) +
                f"\n\nLet me assess what we can do about {'these issues' if issue_count > 1 else 'this issue'}! 🚴‍♂️"
    )

    return {
        **state,
        "collected_info": structured_dict,
        "messages": state["messages"] + [success_msg],
    }


def route_from_issue_classifier(state: State) -> str:
    collected = state.get("collected_info", {})
    bike_type = collected.get("bike_type")
    issues = collected.get("issues", [])

    # If no data collected at all, continue gathering
    if not collected or not bike_type or not issues:
        return "audio_intro"

    # Check if bike_type is valid
    if bike_type == "NULL" or bike_type not in POSSIBLE_VALUES.get('bike_type', []):
        return "audio_intro"

    # Check if ALL issues are complete
    all_issues_complete = True
    for issue in issues:
        required_fields = ["part_category", "part_name", "likely_service"]
        if any(issue.get(field) in [None, "null", "", "NULL"] for field in required_fields):
            all_issues_complete = False
            break

    # Only proceed to repair assessment if everything is complete
    if all_issues_complete:
        print(f"[DEBUG] All information complete. Proceeding to repair assessment.")
        return "repair_assessment"
    else:
        print(f"[DEBUG] Incomplete information detected. Requesting more details.")
        return "audio_intro"


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

def handle_confirmation_node(state: State) -> str:
    """Dummy function for testing - always proceeds to repair assessment"""
    print("[DEBUG] In handle_confirmation_node - proceeding to repair assessment")
    return "repair_assessment"


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

# Add edges - simplified back to original flow
graph.set_entry_point("audio_intro")
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

# Compile the graph
app = graph.compile(checkpointer=memory)


# Helper function to run the system
def run_bike_issue_reporter():
    """
    Main function to run the enhanced bike problem reporter
    """
    print("🚴‍♂️ Enhanced Multi-Issue Bike Problem Reporter Started!")
    print("=" * 60)

    config = {"configurable": {"thread_id": "bike_issues_enhanced"}}

    try:
        # Start the conversation
        initial_input = {
            "messages": [HumanMessage(content="I want to report bike problems")],
            "conversation_round": 0
        }

        # Run the graph
        for step in app.stream(initial_input, config):
            for node_name, node_output in step.items():
                print(f"\n[{node_name.upper()}]")
                if "messages" in node_output and node_output["messages"]:
                    latest_message = node_output["messages"][-1]
                    print(f"💬 {latest_message.content}")

        print("\n" + "=" * 60)
        print("✅ Multi-Issue Bike Problem Reporter Session Complete!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_bike_issue_reporter()