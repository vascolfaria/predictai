import json
import base64
import os

PARTS_FILE = "./parts.json"

def encode_image(image_path):
    """Encode image to base64 string"""
    try:
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found: {image_path}")
            return None
        
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to encode image {image_path}: {e}")
        return None

def display_issues(issues):
    """Generate markdown with embedded images for identified parts"""
    if not isinstance(issues, list):
        issues = [issues]  # Wrap single dict in list if needed

    try:
        with open(PARTS_FILE) as f:
            parts = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Parts file not found: {PARTS_FILE}")
        return "❌ Parts database not found."
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in parts file: {PARTS_FILE}")
        return "❌ Parts database is corrupted."

    markdown = "🧩 Identified Part(s):\n\n"
    
    for issue in issues:
        part_name = issue.get("part_name")
        if not part_name:
            continue
            
        print(f"[DEBUG] Looking for part: {part_name}")
        
        # Find matching part (case-insensitive)
        matching = next((p for p in parts if p["name"].lower() == part_name.lower()), None)

        if matching:            
            markdown += f"⚙ {matching['name']}\n"
            markdown += f"{matching['description']}\n\n"
            
        else:
            print(f"[WARNING] No match found for part: {part_name}")
            markdown += f"⚙ {part_name}\n"
            markdown += "_Part information not available._\n\n"

    return markdown