import json
import os
import base64

PARTS_FILE = "data/parts.json"

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def display_issues(issues):
    with open(PARTS_FILE) as f:
        parts = json.load(f)

    markdown = "### 🧩 Identified Parts:\n\n"
    for issue in issues:
        part_name = issue.get("part_name")
        matching = next((p for p in parts if p["name"].lower() == part_name.lower()), None)
        if matching:
            img_path = matching["image_url"]
            img_base64 = encode_image(img_path)
            markdown += f"#### {matching['name']}\n"
            markdown += f"{matching['description']}\n\n"
            markdown += f"![{matching['name']}](data:image/png;base64,{img_base64})\n\n"
        else:
            markdown += f"#### {part_name}\n"
            markdown += "_No image available._\n\n"

    return markdown