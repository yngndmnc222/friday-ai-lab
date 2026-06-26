import streamlit as st

# Set the page configuration
st.set_page_config(page_title="Chatbot UI", layout="wide")

# Set the background color
st.markdown("<style>body {background-color: lightblue;}</style>", unsafe_allow_html=True)

# Create a title
st.title("Chatbot Interface")

# Create a round text box for user input
user_input = st.text_input("Type your message:", placeholder="Ask me anything...", key="input")

# Display the user input (for demonstration, replace with actual chatbot logic)
if user_input:
    st.write(f"User: {user_input}")
    # Here you would call your chatbot logic to get a response
    # For now, we will just echo the user's message back
    st.write(f"Chatbot: You said '{user_input}'")
