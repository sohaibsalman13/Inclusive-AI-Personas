import json
import subprocess
import os

# --- CONFIGURATION ---
JSON_FILE = "personas.json"
MODEL_NAME = "sauerkraut"


def load_personas():
    """Reads the external JSON file safely."""
    if not os.path.exists(JSON_FILE):
        print(f"Error: Could not find '{JSON_FILE}'")
        return None

    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f" Error: Your '{JSON_FILE}' has a formatting mistake.")
        return None


def ask_ollama(model, prompt):
    """Sends prompt to Ollama."""
    try:
        process = subprocess.Popen(
            ['ollama', 'run', model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt)
        return stdout.strip()
    except Exception as e:
        return f"Error connecting to Ollama: {e}"


def main():
    personas = load_personas()
    if not personas:
        return

    # --- CHOOSE CHARACTER ---
    print("\n AI PERSONAS CHATBOT")
    print(f"Loaded {len(personas)} personas from {JSON_FILE}\n")

    for i, s in enumerate(personas):
        print(f"[{i + 1}] {s['name']} - {s['education']['major']}")

    choice = input("\nEnter number: ")

    try:
        selected = personas[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Exiting.")
        return

    print(f"\n Chatting with: {selected['name']}")
    print(f" From: {selected['demographics']['location']}")
    print("(Type 'exit' to quit)\n")

    # --- START CHAT ---
    chat_history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # --- BUILD THE DEEP PROMPT ---
        prompt = f"""
            Act strictly as this specific student in Munich.

            ### YOUR PROFILE:
            - Name: {selected['name']}
            - Age/Gender: {selected['demographics']['age']}, {selected['demographics']['gender']}
            - Social Class: {selected['demographics']['social_class']}
            - Major: {selected['education']['major']} ({selected['education']['university']})
            - Political Leaning: {selected['psychographics']['political_leaning']}
            - Core Values: {", ".join(selected['psychographics']['values'])}
            - Deepest Belief: "{selected['psychographics']['beliefs']}"
            - Secret Fear: "{selected['psychographics']['fears']}"
            - Typical Evening: {selected['lifestyle']['evening_activity']}

            ### CONVERSATION HISTORY:
            {chat_history}

            ### CURRENT MESSAGE:
            User: {user_input}
            You (Reply in character, Keep it short & conversational):
            """

        print("... (Thinking) ...")

        response = ask_ollama(MODEL_NAME, prompt)

        # CHECK FOR EMPTY RESPONSES
        if not response:
            print(" Error: The model returned nothing. Is Ollama running?")
            continue

        # Clean up output (remove "Name:" if the model accidentally types it)
        clean_response = response.replace(f"{selected['name']}:", "").strip()

        print(f"{selected['name']}: {clean_response}")

        chat_history += f"User: {user_input}\nYou: {clean_response}\n"


if __name__ == "__main__":
    main()