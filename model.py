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
        return []

def get_response(selected, user_input, chat_history):
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

        print("(thinking)")

        try:
            response = ollama.chat(model=MODEL_NAME, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"Error: {e}"


