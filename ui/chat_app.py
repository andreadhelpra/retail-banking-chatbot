import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="BNP Paribas AI Assistant",
    page_icon="🏦",
    layout="wide",
)

# Custom CSS for BNP branding
st.markdown(
    """
    <style>
    .main-header {
        color: #00915A;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .debug-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .debug-label {
        font-weight: 600;
        color: #00915A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session():
    """Initialize a chat session with the backend."""
    try:
        with httpx.Client() as client:
            resp = client.post(f"{API_BASE_URL}/session/init")
            resp.raise_for_status()
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.customer = data["customer"]
    except httpx.HTTPError as e:
        st.error(f"Could not connect to backend: {e}. Make sure the FastAPI server is running.")
        st.stop()


def send_message(message: str) -> dict:
    """Send a message to the chat endpoint."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{API_BASE_URL}/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": message,
            },
        )
        resp.raise_for_status()
        return resp.json()


# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.debug_history = []
    st.session_state.session_id = None
    st.session_state.customer = None

if st.session_state.session_id is None:
    init_session()

# --- Sidebar: Debug Panel ---
with st.sidebar:
    st.markdown("### Debug Panel")
    st.markdown("---")

    if st.session_state.customer:
        st.markdown("**Authenticated as:**")
        st.write(st.session_state.customer["name"])
        st.markdown("**Accounts:**")
        for acc in st.session_state.customer["accounts"]:
            st.write(f"- {acc['label']} ({acc['type']})")
        st.markdown("**Cards:**")
        for card in st.session_state.customer["cards"]:
            st.write(f"- {card['label']} (****{card['last_four']}, {card['status']})")
        st.markdown("---")

    # Show debug info for the most recent exchange
    if st.session_state.debug_history:
        latest = st.session_state.debug_history[-1]
        st.markdown("**Latest Request Debug:**")
        st.markdown(f"**Agent:** `{latest.get('agent', 'N/A')}`")
        st.markdown(f"**Intent:** `{latest.get('intent', 'N/A')}`")
        st.markdown(f"**Confidence:** `{latest.get('confidence', 'N/A')}`")

        entities = latest.get("entities", {})
        if entities:
            st.markdown("**Entities:**")
            st.json(entities)

        chunks = latest.get("retrieved_chunks")
        if chunks:
            st.markdown("**Retrieved FAQ Chunks:**")
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i + 1}"):
                    st.text(chunk[:500])

        tool_calls = latest.get("tool_calls")
        if tool_calls:
            st.markdown("**Tool Calls:**")
            st.json(tool_calls)

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.debug_history = []
        st.session_state.session_id = None
        st.session_state.customer = None
        st.rerun()

# --- Main Chat Area ---
st.markdown('<div class="main-header">BNP Paribas AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retail Banking Support — Demo Prototype</div>', unsafe_allow_html=True)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to backend and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = send_message(prompt)
                response = result["response"]
                debug = result.get("debug", {})

                st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.debug_history.append(debug)
            except httpx.HTTPError as e:
                error_msg = f"Error communicating with backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.rerun()
