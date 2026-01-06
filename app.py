import streamlit as st
import model

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Personas Bot", page_icon="🎓", layout="wide")
st.title("AI Personas Chatbot")

# --- SIDEBAR: SELECT STUDENT ---
students = model.load_students()

if not students:
    st.error("Could not load personas.json! Make sure the file is in the same folder.")
    st.stop()

st.sidebar.header("Select Persona")

# Create a list of names for the dropdown
student_names = [s['name'] for s in students]
selected_index = st.sidebar.selectbox("Choose a persona:", range(len(student_names)),
                                      format_func=lambda x: student_names[x])
selected_persona = students[selected_index]

# Display Student Details in Sidebar
st.sidebar.divider()
st.sidebar.subheader(f" {selected_persona['name']}")
st.sidebar.write(f"**Age:** {selected_persona['demographics']['age']}")
st.sidebar.write(f"**Origin:** {selected_persona['demographics']['origin']}")

# --- CHAT INTERFACE ---

# 1. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Reset Chat if User Changes Persona
if "current_persona_name" not in st.session_state:
    st.session_state.current_persona_name = selected_persona['name']

if st.session_state.current_persona_name != selected_persona['name']:
    st.session_state.messages = []  # Wipe history
    st.session_state.current_persona_name = selected_persona['name']

# 3. Display Old Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle New Input
if user_input := st.chat_input("Type your message..."):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save User Message to History
    st.session_state.messages.append({"role": "user", "content": user_input})

    # --- THE BACKEND CALL ---
    with st.chat_message("assistant"):
        with st.spinner(f"{selected_persona['name']} is thinking..."):
            # Call the function from model.py!
            bot_reply = model.get_response(
                selected_persona,
                user_input,
                st.session_state.messages
            )

            st.markdown(bot_reply)

    # Save Bot Message to History
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})