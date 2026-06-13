import random
import string
import urllib.error
import urllib.request
import cv2
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
# Assuming deepagents is your local or specific custom framework package
from deepagents import create_deep_agent

# Global or state-based dictionary to hold the active image in memory
# since LLMs cannot easily pass raw numpy arrays back and forth in JSON.
CURRENT_IMAGE_STATE = {"image": None}

@tool
def search_image_from_web(description: str) -> str:
    """
    Searches the web for an image based on a description and loads it.
    Returns a success confirmation message.
    """
    # Placeholder: In a production app, use an image search API.
    # For demonstration, we create a dummy placeholder image (blank canvas).
    blank_image = np.zeros((500, 500, 3), dtype=np.uint8) + 128  # Gray canvas
    CURRENT_IMAGE_STATE["image"] = blank_image
    return f"Successfully found and loaded image for: '{description}'."

@tool
def create_meme(text: str) -> str:
    """
    Generates a meme by overlaying the provided text onto the loaded image.
    If the text contains punctuation (like a period, comma, or exclamation),
    it splits it into top and bottom text. Otherwise, it places it at the bottom.
    """
    img = CURRENT_IMAGE_STATE.get("image")
    if img is None:
        return "Error: No image has been searched or loaded yet."

    # Split text if punctuation exists
    delimiters = [".", ",", "!", "?", ";"]
    split_char = None
    for char in delimiters:
        if char in text:
            split_char = char
            break

    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (255, 255, 255) # White
    thickness = 2

    if split_char:
        parts = text.split(split_char, 1)
        top_text = parts[0].strip()
        bottom_text = parts[1].strip() if parts[1].strip() else top_text

        # Draw Top Text
        cv2.putText(img, top_text, (20, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        # Draw Bottom Text
        cv2.putText(img, bottom_text, (20, h - 30), font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        # Draw Bottom Text only
        cv2.putText(img, text.strip(), (20, h - 30), font, font_scale, color, thickness, cv2.LINE_AA)

    CURRENT_IMAGE_STATE["image"] = img
    return "Meme text successfully applied to the image."

@tool
def save_meme(file_name: str = "") -> str:
    """
    Saves the generated meme to disk. If no file name is provided,
    a random sequence of characters is generated as the file name.
    Forces .png format.
    """
    img = CURRENT_IMAGE_STATE.get("image")
    if img is None:
        return "Error: No meme image exists to save."

    if not file_name or file_name.strip() == "":
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        file_name = f"{random_str}.png"

    if not file_name.lower().endswith('.png'):
        file_name = f"{file_name.split('.')[0]}.png"

    cv2.imwrite(file_name, img)
    return f"Meme successfully saved as '{file_name}'."


SYSTEM_PROMPT = """You are a meme generator agent.

## Capabilities
- 'search_image_from_web': Searches the web for image based on description and loads the first image it finds to the conversation.
- 'create_meme': Generates meme from searched image and provided text. If the text has punctuation, the first part of the text is written at the top of the image and the second part at the bottom of the image. If not, then the text is written at the bottom of the image. All the texts should be on the image.
- 'save_meme': Saves generated meme. If no image name is generated, a random sequence of characters and letters is given to as file name. The saved image should be of .png format.

Do not assume anything and follow the instructions as it is. Always call the tools in the correct logical sequence: search -> create -> save.
"""

# Initialize Model
model = init_chat_model(
    "gemini-3.1-pro-preview",
    model_provider="google-genai",
    temperature=0.5,
    timeout=600,
    max_tokens=25000,
    streaming=True,
)

checkpointer = InMemorySaver()

# Initialize Agent with ALL required tools
deep_agent = create_deep_agent(
    model=model,
    tools=[search_image_from_web, create_meme, save_meme],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = """The image is of a lady athlete competing in running.

The meme text is:

A fast sandwich maker.
"""

# Invoke Agent
deep_agent_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "meme-gen"}},
)
