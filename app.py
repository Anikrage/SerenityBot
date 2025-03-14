from flask import Flask, request, jsonify, render_template
import ollama
import xml.etree.ElementTree as ET
import os
import re

app = Flask(__name__)
MODEL_NAME = "deepseek-r1:1.5b"
MEMORY_FILE = "conversation_memory.xml"


def load_memory():
    """Load conversation memory from XML file or create a new one if it doesn't exist."""
    if not os.path.exists(MEMORY_FILE):
        root = ET.Element("conversation")
        tree = ET.ElementTree(root)
        tree.write(MEMORY_FILE)
    return ET.parse(MEMORY_FILE)


def save_message(role, content):
    """Save a message to the conversation memory."""
    tree = load_memory()
    root = tree.getroot()
    msg = ET.SubElement(root, "message")
    ET.SubElement(msg, "role").text = role
    ET.SubElement(msg, "content").text = content
    tree.write(MEMORY_FILE)


def get_context():
    """Retrieve the last 5 messages from the conversation memory to keep responses focused."""
    tree = load_memory()
    return "\n".join(
        f"{msg.find('role').text}: {msg.find('content').text}"
        for msg in tree.findall("message")[-5:]  # Reduce context size for relevance
    )


def remove_think_tags(text):
    """Remove content within <think> tags from the text."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def refine_response(response):
    """Post-process response to limit length and improve relevance."""
    response = remove_think_tags(response)
    sentences = response.split(". ")  # Split into sentences
    short_response = ". ".join(sentences[:2])  # Keep only the first 2 sentences
    return short_response.strip()


@app.route('/')
def home():
    """Render the chatbot interface."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle user messages and generate chatbot responses."""
    user_input = request.json['message']
    save_message("User", user_input)
    
    context = get_context()
    
    # Generate a response using the Ollama model with length constraint
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=f"{context}\nTherapist (concise and relevant reply only, max 2 sentences):",
    )['response'].strip()
    
    cleaned_response = refine_response(response)
    
    # Save and return the cleaned response
    save_message("Therapist", cleaned_response)
    return jsonify({'response': cleaned_response})


if __name__ == '__main__':
    app.run(debug=True)
