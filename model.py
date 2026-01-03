import json
import ollama

# --- CONFIGURATION ---
MODEL_NAME = "hf.co/MaziyarPanahi/Llama-3-SauerkrautLM-8b-Instruct-GGUF"
JSON_FILE = "personas.json"


def load_personas():
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f" Error loading JSON: {e}")
        return None


def main():
    personas = load_personas()
    if not personas: return

    # --- MENU ---
    print("\n🎓 MUNICH STUDENT CHATBOT 🎓")
    for i, s in enumerate(personas):
        print(f"[{i + 1}] {s['name']} - {s['education']['major']}")

    choice = input("\nEnter number: ")
    try:
        selected = personas[int(choice) - 1]
    except:
        return

    print(f"\n Chatting with: {selected['name']}")
    chat_history = []

    # --- CHAT LOOP ---
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]: break

        # We inject the Persona Data directly into the prompt (No LoRA needed!)
        system_instruction = f"""
        Du bist ein Student in München. Antworte immer auf Deutsch.
        Bleib in deiner Rolle!

        DEIN PROFIL:
        - Name: {selected['name']}
        - Alter: {selected['demographics']['age']}
        - Studium: {selected['education']['major']} an der {selected['education']['university']}
        - Wohnort: {selected['demographics']['location']} ({selected['demographics']['social_class']})
        - Hobbys: {", ".join(selected['lifestyle']['hobbies'])}
        - Dein Glaube: "{selected['psychographics']['beliefs']}"
        - Deine Sprache: Nutze Jugendsprache, "Digga", "Alter", oder Münchner Dialekt wenn passend.
        """

        # Build message history for the model
        messages = [{'role': 'system', 'content': system_instruction}]
        messages.extend(chat_history)
        messages.append({'role': 'user', 'content': user_input})

        print("...")

        # Send to Ollama
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        answer = response['message']['content']

        print(f"{selected['name']}: {answer}")

        # Update history
        chat_history.append({'role': 'user', 'content': user_input})
        chat_history.append({'role': 'assistant', 'content': answer})


if __name__ == "__main__":
    main()