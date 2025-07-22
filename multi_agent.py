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
from utils_functions.show_image import display_issues
from pydantic import BaseModel

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
        # Fall back to asking for more information
        new_round = state.get("conversation_round", 0) + 1
        return {
            **state,
            "conversation_round": new_round,
            "messages": state["messages"] + [
                AIMessage(
                    content="I need a bit more information to help you properly. Could you tell me:\n"
                            "1. What type of bike you have\n"
                            "2. Which specific parts have problems\n"
                            "3. What exactly is wrong with each part?")
            ],
            "current_agent": "audio_intro"
        }

    # Check if we have enough information
    bike_type = structured_dict.get('bike_type')
    issues = structured_dict.get('issues', [])

    if not bike_type or bike_type == 'NULL' or not issues:
        new_round = state.get("conversation_round", 0) + 1
        return {
            **state,
            "conversation_round": new_round,
            "messages": state["messages"] + [
                AIMessage(
                    content="Thanks! I still need more details about your bike type and the specific issues. Can you provide more information?")
            ],
            "current_agent": "audio_intro"
        }

    # Check if individual issues are complete
    incomplete_issues = []
    for i, issue in enumerate(issues):
        required_fields = ["part_category", "part_name", "likely_service"]
        missing = [field for field in required_fields if issue.get(field) in [None, "null", "", "NULL"]]
        if missing:
            incomplete_issues.append(f"Issue {i + 1}: {', '.join(missing)}")

    if incomplete_issues:
        new_round = state.get("conversation_round", 0) + 1
        missing_info = "; ".join(incomplete_issues)
        return {
            **state,
            "conversation_round": new_round,
            "messages": state["messages"] + [
                AIMessage(
                    content=f"Great start! I still need some details: {missing_info}. Can you provide more information about these specific problems?")
            ],
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

    if not bike_type or not issues:
        return "audio_intro"

    # Check if all issues have required fields
    for issue in issues:
        required_fields = ["part_category", "part_name", "likely_service"]
        if any(issue.get(field) in [None, "null", "", "NULL"] for field in required_fields):
            return "audio_intro"

    return "repair_assessment"


def wait_for_human_node(state: State) -> State:
    return state


# Build LangGraph - back to original structure
graph = StateGraph(State)

# Add nodes
graph.add_node("audio_intro", audio_intro_node)
graph.add_node("audio_record", audio_record_node)
graph.add_node("audio_process", audio_process_node)
graph.add_node("issue_classifier", issue_classifier_node)
graph.add_node("wait_for_human", wait_for_human_node)
graph.add_node("repair_assessment", repair_assessment_node)

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
        "repair_assessment": "repair_assessment"
    }
)
graph.add_edge("repair_assessment", END)

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