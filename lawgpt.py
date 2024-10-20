import openai
import streamlit as st
openai.api_key = "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"
st.set_page_config(page_title="Bharat Legal GPT")
file_handler = st.container()
st.title("⚖️ Legal BOT")
st.write('''Celebrate Legal Empowerment with Legal BOT: Your Trusted Partner for Instant Legal Clarity and Expert Guidance – Making Law Simple and Accessible for Everyone!''')
st.markdown('\n')
st.markdown('\n')
with file_handler:
    if st.button("🔃 Refresh"):
        st.cache_data.clear()
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Define conversation examples
conversations = [
    {
        "messages": [
            {"role": "system", "content": "Welcome to the Legal BOT- Your Expert in Indian Legal Matters!"},
            {"role": "user", "content": "What can you do?"},
            {"role": "assistant", "content": "Legal BOTcan assist you with a wide range of Indian legal matters. You can ask me questions about starting a business, patent filing, labor laws, contracts, and more. I can provide information, guidance, and legal insights about different sections of the Indian Penal Code (IPC) , criminal procedure, offenses, or legal proceedings related to criminal matters, to help you navigate the complex legal landscape in India. Feel free to ask any legal questions, and I'll do my best to provide you with accurate and helpful answers."},
        ]
    }
]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
full_response = ""
if prompt := st.chat_input("Enter a prompt here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Append the user's message to the conversation
    conversations[-1]["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        responses = ""
        for response in openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversations[-1]["messages"],
            stream=True,
        ):
            if response is not None:
                if response.choices[0].delta.content is not None:
                    responses = response.choices[0].delta.content
                    if responses:
                        full_response += responses
                        message_placeholder.markdown(full_response + "▌")
                else:
                    break
            else:
                break
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})