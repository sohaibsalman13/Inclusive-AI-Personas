import json
import ollama
import sys

# --- CONFIGURATION ---
MODEL_NAME = "sauerkraut"
JSON_FILE = "personas.json"


def load_students():
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f" Error: Could not find '{JSON_FILE}'.")
        return None
    except json.JSONDecodeError:
        print(f" Error: '{JSON_FILE}' is not valid JSON.")
        return None


def main():
    students = load_students()
    if not students: return

    # --- MENU ---
    print("\n MUNICH STUDENT CHATBOT ")

    # Display the menu using the nested JSON structure
    for i, s in enumerate(students):
        print(f"[{i + 1}] {s['name']} - {s['education']['major']} ({s['demographics']['social_class']})")

    # --- SELECTION ---
    try:
        choice_input = input("\nEnter number (or 'q' to quit): ")
        if choice_input.lower() == 'q': return

        selected = students[int(choice_input) - 1]
    except (ValueError, IndexError):
        print(" Invalid selection. Exiting.")
        return

    print(f"\n Chatting with: {selected['name']}")
    print(f" Location: {selected['demographics']['location']}")
    print(f" Core Value: {selected['psychographics']['values'][0]}")
    print("(Type 'exit' to quit)\n")

    chat_history = []

    # --- CHAT LOOP ---
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]: break

        # --- DYNAMIC SYSTEM PROMPT ---
        system_instruction = f"""
        Du bist jetzt diese Person. Antworte immer auf Deutsch.
        Bleibe strikt in deiner Rolle! Brich nicht aus deiner Rolle aus.

        ### DEIN PROFIL:
        - Name: {selected['name']}
        - Alter/Geschlecht: {selected['demographics']['age']}, {selected['demographics']['gender']}
        - Herkunft: {selected['demographics']['origin']} ({selected['demographics']['social_class']})
        - Wohnort: {selected['demographics']['location']}

        ### AUSBILDUNG:
        - Studienfach: {selected['education']['major']} an der {selected['education']['university']} ({selected['education']['status']})

        ### DEINE PSYCHE (WICHTIG):
        - Werte: {", ".join(selected['psychographics']['values'])}
        - Glaube/Religion: {selected['psychographics']['religion']}
        - Tiefe Überzeugungen: "{selected['psychographics']['beliefs']}"
        - Deine Ängste: "{selected['psychographics']['fears']}"

        ### LEBENSSTIL:
        - Hobbys: {", ".join(selected['lifestyle']['hobbies'])}
        - Typischer Abend: {selected['lifestyle']['evening_activity']}
        - Transportmittel: {selected['lifestyle']['transport']}

        ### SPRACHSTIL:
        Sprich wie eine echte Person, die in München lebt.
        Nutze eine Sprache, die deinem Alter und deinen Überzeugungen entspricht.
        Antworte kurz und direkt.
        """

        # Build the message chain
        messages = [{'role': 'system', 'content': system_instruction}]
        messages.extend(chat_history)
        messages.append({'role': 'user', 'content': user_input})

        print("... (thinking)")

        try:
            # Send to Ollama
            response = ollama.chat(model=MODEL_NAME, messages=messages)
            answer = response['message']['content']

            # Print output
            print(f"{selected['name']}: {answer}")

            # Update history
            chat_history.append({'role': 'user', 'content': user_input})
            chat_history.append({'role': 'assistant', 'content': answer})

        except Exception as e:
            print(f" Error: {e}")
            break


if __name__ == "__main__":
    main()


