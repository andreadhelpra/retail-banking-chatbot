"""BNP Paribas AI Assistant — Production Chat Interface"""

import html
import json
from pathlib import Path

import httpx
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"
LOGO_PATH = Path(__file__).parent / "static" / "logo-bnp.svg"

# ─── Page Setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BNP Paribas — AI Assistant",
    page_icon=str(Path(__file__).parent / "static" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,400,0,0" rel="stylesheet">

<style>
/* ── Variables ──────────────────────────────────────────────── */
:root {
    --bnp-green: #00915A;
    --bnp-green-dark: #006B43;
    --bnp-green-deep: #004D31;
    --bnp-green-light: #7FCBAE;
    --bnp-green-pale: #E8F5EF;
    --bnp-green-ghost: #F2FAF6;
    --bnp-black: #1A1A2E;
    --bnp-gray-900: #2D2D3F;
    --bnp-gray-700: #4A4A5A;
    --bnp-gray-600: #6B7280;
    --bnp-gray-400: #9CA3AF;
    --bnp-gray-300: #D1D5DB;
    --bnp-gray-200: #E5E7EB;
    --bnp-gray-100: #F3F4F6;
    --bnp-gray-50: #F9FAFB;
    --bnp-white: #FFFFFF;
    --radius-sm: 8px;
    --radius: 12px;
    --radius-lg: 16px;
    --font: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Global ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
}

/* Top brand bar */
.main::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #008053, #00A76D, #7FCBAE, #00A76D, #008053);
    z-index: 9999;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3px !important;
    min-height: 3px !important;
}

.main .block-container {
    padding: 1.5rem 2rem 0 2rem;
    max-width: 100%;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bnp-white);
    border-right: 1px solid var(--bnp-gray-200);
    width: 280px !important;
}

/* Logo */
.sidebar-logo {
    padding: 0 1rem 0.25rem 1rem;
}

.sidebar-logo svg {
    max-width: 160px;
    height: auto;
    display: block;
}

[data-testid="stSidebar"] hr {
    margin: 0.75rem 1rem;
    border-color: var(--bnp-gray-200);
}

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    font-family: var(--font) !important;
    font-weight: 500 !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}

/* Navigation */
.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 0.75rem;
}

.sidebar-nav .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    color: var(--bnp-gray-700);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
    font-family: var(--font);
    transition: all 0.15s ease;
    cursor: pointer;
    border: none;
    background: none;
    width: 100%;
    text-align: left;
}

.sidebar-nav .nav-item:hover {
    background: var(--bnp-green-ghost);
    color: var(--bnp-green);
}

.sidebar-nav .nav-item.active {
    background: var(--bnp-green-pale);
    color: var(--bnp-green);
    font-weight: 600;
}

.sidebar-nav .nav-item .material-symbols-outlined {
    font-size: 20px;
    line-height: 1;
}

.sidebar-nav .nav-item.active .material-symbols-outlined {
    font-variation-settings: 'FILL' 1;
}

.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--bnp-gray-400);
    padding: 12px 14px 6px 14px;
    font-family: var(--font);
    margin: 0 0.75rem;
}

/* External link */
.sidebar-external {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    margin: 0 0.75rem;
    border-radius: var(--radius-sm);
    color: var(--bnp-gray-400);
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 500;
    font-family: var(--font);
    transition: all 0.15s ease;
}

.sidebar-external:hover {
    color: var(--bnp-green);
    background: var(--bnp-green-ghost);
}

.sidebar-external .material-symbols-outlined {
    font-size: 16px;
}

/* User profile */
.user-profile {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    margin: 0 0.75rem;
    border-radius: var(--radius);
    background: var(--bnp-gray-50);
    border: 1px solid var(--bnp-gray-200);
}

.user-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #008053, #00A76D);
    color: var(--bnp-white);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    flex-shrink: 0;
    font-family: var(--font);
}

.user-details {
    flex: 1;
    min-width: 0;
}

.user-name {
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--bnp-black);
    font-family: var(--font);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.user-role {
    font-size: 0.75rem;
    color: var(--bnp-gray-400);
    font-family: var(--font);
    margin-top: 1px;
}

/* ── Chat Area ──────────────────────────────────────────────── */
.chat-header-fixed {
    position: fixed;
    top: 3px;
    left: 280px;
    right: 0;
    z-index: 500;
    background: var(--bnp-white);
    border-bottom: 1px solid var(--bnp-gray-200);
    padding: 1rem 2rem;
}

.chat-header {
    max-width: 760px;
    margin: 0 auto;
}

/* Spacer to prevent content from hiding behind fixed header */
.chat-header-spacer {
    height: 72px;
}

.chat-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.chat-title {
    font-family: var(--font);
    font-weight: 700;
    font-size: 1.35rem;
    color: var(--bnp-black);
    letter-spacing: -0.02em;
    margin: 0;
}

.chat-subtitle {
    font-family: var(--font);
    font-size: 0.825rem;
    color: var(--bnp-gray-400);
    margin: 2px 0 0 0;
    font-weight: 400;
}

.chat-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    background: var(--bnp-green-pale);
    color: var(--bnp-green);
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: var(--font);
}

.chat-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--bnp-green);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Messages ──────────────────────────────────────────────── */

/* User message — right-aligned bubble */
.user-msg-row {
    display: flex;
    justify-content: flex-end;
    padding: 6px 0;
    max-width: 760px;
    margin: 0 auto;
    animation: fadeInUp 0.2s ease-out;
}

.user-msg-bubble {
    background: var(--bnp-green-pale);
    color: var(--bnp-green-deep);
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.925rem;
    line-height: 1.65;
    font-weight: 500;
    font-family: var(--font);
    word-wrap: break-word;
}

/* Assistant message — full-width plain text */
.assistant-msg {
    max-width: 760px;
    margin: 0 auto;
    padding: 6px 0 2px 0;
    animation: fadeInUp 0.2s ease-out;
}

/* Style markdown rendered by st.markdown inside the assistant flow */
.main .block-container [data-testid="stMarkdown"] p,
.main .block-container [data-testid="stMarkdown"] li {
    font-family: var(--font) !important;
    font-size: 0.925rem !important;
    line-height: 1.65 !important;
    color: var(--bnp-gray-900) !important;
}

/* ── Tool indicator expander ───────────────────────────────── */

/* Top-level expanders in main area = tool indicators */
.main [data-testid="stExpander"] {
    background: var(--bnp-green-ghost) !important;
    border: 1px solid var(--bnp-green-pale) !important;
    border-left: 3px solid var(--bnp-green) !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
    margin-top: 4px !important;
    margin-bottom: 16px !important;
    max-width: 760px;
}

.main [data-testid="stExpander"] summary {
    font-family: var(--font) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--bnp-green-dark) !important;
    padding: 10px 14px !important;
}

.main [data-testid="stExpander"] summary p,
.main [data-testid="stExpander"] summary span {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--bnp-green-dark) !important;
    line-height: 1.4 !important;
}

.main [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: var(--bnp-white) !important;
    border-top: 1px solid var(--bnp-green-pale) !important;
    padding: 12px !important;
}

/* Nested expanders (sources inside tool expander) */
.main [data-testid="stExpander"] [data-testid="stExpander"] {
    background: var(--bnp-white) !important;
    border: 1px solid var(--bnp-gray-200) !important;
    border-left: 2px solid var(--bnp-gray-300) !important;
    margin: 4px 0 !important;
}

.main [data-testid="stExpander"] [data-testid="stExpander"] summary {
    color: var(--bnp-gray-600) !important;
    font-weight: 500 !important;
}

/* Expander detail text */
.main [data-testid="stExpanderDetails"] p,
.main [data-testid="stExpanderDetails"] span,
.main [data-testid="stExpanderDetails"] li {
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}

/* Debug detail labels */
.debug-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px 16px;
    align-items: baseline;
    font-family: var(--font);
    font-size: 0.8rem;
    padding: 4px 0;
}

.debug-label {
    font-weight: 600;
    color: var(--bnp-gray-600);
    white-space: nowrap;
}

.debug-value {
    color: var(--bnp-black);
    font-weight: 500;
}

.debug-value code {
    background: var(--bnp-green-pale);
    color: var(--bnp-green-dark);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.debug-chunk {
    background: var(--bnp-gray-50);
    border: 1px solid var(--bnp-gray-200);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-size: 0.78rem;
    line-height: 1.55;
    color: var(--bnp-gray-700);
    font-family: var(--font);
    margin: 4px 0;
    max-height: 150px;
    overflow-y: auto;
}

.debug-json {
    background: var(--bnp-gray-900);
    color: #a8d8a8;
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-size: 0.75rem;
    line-height: 1.5;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    margin: 4px 0;
    overflow-x: auto;
    max-height: 250px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Confidence bar */
.confidence-bar {
    display: flex;
    align-items: center;
    gap: 8px;
}

.confidence-track {
    flex: 1;
    height: 4px;
    background: var(--bnp-gray-200);
    border-radius: 2px;
    overflow: hidden;
    max-width: 120px;
}

.confidence-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--bnp-green);
    transition: width 0.3s ease;
}

.confidence-text {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--bnp-green-dark);
    font-family: var(--font);
}

/* Chat input */
[data-testid="stChatInput"] {
    border-color: var(--bnp-gray-300) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--bnp-green) !important;
    box-shadow: 0 0 0 3px rgba(0, 145, 90, 0.1) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: var(--font) !important;
    font-size: 0.925rem !important;
}

/* ── Welcome Screen ─────────────────────────────────────────── */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem 2rem 2rem;
    text-align: center;
}

.welcome-icon {
    width: 64px;
    height: 64px;
    border-radius: 20px;
    background: linear-gradient(135deg, #008053, #00A76D);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(0, 145, 90, 0.2);
}

.welcome-icon .material-symbols-outlined {
    font-size: 32px;
    color: white;
}

.welcome-title {
    font-family: var(--font);
    font-weight: 700;
    font-size: 1.5rem;
    color: var(--bnp-black);
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
}

.welcome-subtitle {
    font-family: var(--font);
    font-weight: 400;
    font-size: 0.95rem;
    color: var(--bnp-gray-400);
    max-width: 420px;
    line-height: 1.5;
    margin: 0 0 2rem 0;
}

/* Suggestion buttons override */
.suggestion-btn button {
    width: 100% !important;
    background: var(--bnp-white) !important;
    border: 1px solid var(--bnp-gray-200) !important;
    border-radius: var(--radius) !important;
    padding: 14px 16px !important;
    text-align: left !important;
    font-family: var(--font) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--bnp-gray-700) !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    line-height: 1.4 !important;
}

.suggestion-btn button:hover {
    border-color: var(--bnp-green) !important;
    background: var(--bnp-green-ghost) !important;
    color: var(--bnp-green) !important;
    box-shadow: 0 2px 8px rgba(0, 145, 90, 0.08) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--bnp-gray-300);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--bnp-gray-400);
}

/* ── Animations ─────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── API Helpers ──────────────────────────────────────────────────────────────


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


# ─── Helpers ──────────────────────────────────────────────────────────────────


def get_initials(name: str) -> str:
    parts = name.split()
    return "".join(p[0].upper() for p in parts[:2]) if parts else "?"


def get_debug_summary(debug: dict) -> str:
    """Build a Claude-style 'Used X agent' summary line."""
    agent = (debug.get("agent") or "unknown").replace("_", " ").title()
    parts = [f"Used {agent} agent"]

    tool_calls = debug.get("tool_calls") or []
    if tool_calls:
        n = len(tool_calls)
        parts.append(f"called {n} tool{'s' if n != 1 else ''}")

    chunks = debug.get("retrieved_chunks") or []
    if chunks:
        n = len(chunks)
        parts.append(f"{n} source{'s' if n != 1 else ''}")

    return "  \u00b7  ".join(parts)


def render_debug_details(debug: dict):
    """Render full debug details inside an expander."""
    agent = debug.get("agent", "N/A")
    intent = debug.get("intent", "N/A")
    confidence = debug.get("confidence", 0)

    # Grid of metadata
    conf_pct = int(confidence * 100)
    st.markdown(
        f"""<div class="debug-grid">
            <span class="debug-label">Agent</span>
            <span class="debug-value"><code>{html.escape(str(agent))}</code></span>
            <span class="debug-label">Intent</span>
            <span class="debug-value"><code>{html.escape(str(intent))}</code></span>
            <span class="debug-label">Confidence</span>
            <span class="debug-value">
                <div class="confidence-bar">
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: {conf_pct}%"></div>
                    </div>
                    <span class="confidence-text">{conf_pct}%</span>
                </div>
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Entities
    entities = debug.get("entities", {})
    if entities:
        st.markdown(
            '<div class="debug-label" style="margin-top: 10px; font-size: 0.78rem;">Entities</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="debug-json">{html.escape(json.dumps(entities, indent=2))}</div>',
            unsafe_allow_html=True,
        )

    # Retrieved chunks
    chunks = debug.get("retrieved_chunks") or []
    if chunks:
        st.markdown(
            f'<div class="debug-label" style="margin-top: 10px; font-size: 0.78rem;">Retrieved Sources ({len(chunks)})</div>',
            unsafe_allow_html=True,
        )
        for i, chunk in enumerate(chunks):
            with st.expander(f"Source {i + 1}", expanded=False):
                st.markdown(
                    f'<div class="debug-chunk">{html.escape(chunk[:500])}</div>',
                    unsafe_allow_html=True,
                )

    # Tool calls
    tool_calls = debug.get("tool_calls") or []
    if tool_calls:
        st.markdown(
            f'<div class="debug-label" style="margin-top: 10px; font-size: 0.78rem;">Tool Calls ({len(tool_calls)})</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="debug-json">{html.escape(json.dumps(tool_calls, indent=2))}</div>',
            unsafe_allow_html=True,
        )


def load_logo_svg() -> str:
    """Load logo SVG, stripping XML/DOCTYPE preamble."""
    if not LOGO_PATH.exists():
        return ""
    raw = LOGO_PATH.read_text()
    # Strip everything before <svg to remove XML declarations, DOCTYPE, etc.
    svg_start = raw.find("<svg")
    if svg_start > 0:
        raw = raw[svg_start:]
    return raw


# ─── Initialize Session State ────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.customer = None
    st.session_state.pending_message = None

if st.session_state.session_id is None:
    init_session()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo
    logo_svg = load_logo_svg()
    if logo_svg:
        st.markdown(
            f'<div class="sidebar-logo">{logo_svg}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-logo" style="font-weight:800; font-size:1.1rem; color:#00915A;">BNP PARIBAS</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Navigation
    st.markdown(
        """
        <div class="sidebar-section-label">Banking</div>
        <nav class="sidebar-nav">
            <a class="nav-item active" href="#" onclick="return false;">
                <span class="material-symbols-outlined">smart_toy</span>
                AI Assistant
            </a>
            <a class="nav-item" href="#" onclick="return false;">
                <span class="material-symbols-outlined">account_balance_wallet</span>
                My Accounts
            </a>
            <a class="nav-item" href="#" onclick="return false;">
                <span class="material-symbols-outlined">swap_horiz</span>
                Transfers
            </a>
            <a class="nav-item" href="#" onclick="return false;">
                <span class="material-symbols-outlined">credit_card</span>
                Cards &amp; Payments
            </a>
            <a class="nav-item" href="#" onclick="return false;">
                <span class="material-symbols-outlined">trending_up</span>
                Investments
            </a>
            <a class="nav-item" href="#" onclick="return false;">
                <span class="material-symbols-outlined">help_outline</span>
                Help Center
            </a>
        </nav>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("")

    # External link
    st.markdown(
        """
        <a class="sidebar-external" href="https://group.bnpparibas/en/" target="_blank" rel="noopener">
            <span class="material-symbols-outlined">open_in_new</span>
            group.bnpparibas
        </a>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("---")

    # User profile
    if st.session_state.customer:
        name = st.session_state.customer["name"]
        initials = get_initials(name)
        accounts = st.session_state.customer.get("accounts", [])
        account_label = accounts[0]["type"].title() if accounts else "Personal"

        st.markdown(
            f"""
            <div class="user-profile">
                <div class="user-avatar">{initials}</div>
                <div class="user-details">
                    <div class="user-name">{html.escape(name)}</div>
                    <div class="user-role">{html.escape(account_label)} Banking</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    if st.button("New conversation", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.customer = None
        st.rerun()


# ─── Main Content ────────────────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div class="chat-header-fixed">
        <div class="chat-header">
            <div class="chat-header-row">
                <div>
                    <div class="chat-title">AI Assistant</div>
                    <div class="chat-subtitle">Retail Banking Support</div>
                </div>
                <div class="chat-badge">
                    <span class="chat-badge-dot"></span>
                    Online
                </div>
            </div>
        </div>
    </div>
    <div class="chat-header-spacer"></div>
""",
    unsafe_allow_html=True,
)

# ─── Chat Messages ───────────────────────────────────────────────────────────

# Welcome screen when no messages
if not st.session_state.messages:
    customer_name = ""
    if st.session_state.customer:
        customer_name = st.session_state.customer["name"].split()[0]

    st.markdown(
        f"""
        <div class="welcome-container">
            <div class="welcome-icon">
                <span class="material-symbols-outlined">assistant</span>
            </div>
            <div class="welcome-title">Hello{', ' + html.escape(customer_name) if customer_name else ''}.</div>
            <div class="welcome-subtitle">
                I can help you with your accounts, cards, transfers, and answer
                questions about BNP Paribas services.
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Suggestion buttons
    suggestions = [
        ("Check my balance", "What's my current account balance?"),
        ("Recent transactions", "Show me my recent transactions"),
        ("Card services", "I need help with my card"),
        ("Make a transfer", "I want to make a transfer"),
    ]

    s_cols = st.columns(len(suggestions))
    for col, (label, msg) in zip(s_cols, suggestions):
        with col:
            st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
            if st.button(label, key=f"suggest_{label}", use_container_width=True):
                st.session_state.pending_message = msg
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# Render messages — no st.chat_message, fully custom layout
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        # User: right-aligned bubble
        escaped = html.escape(msg["content"])
        st.markdown(
            f'<div class="user-msg-row"><div class="user-msg-bubble">{escaped}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Assistant: full-width plain text (st.markdown renders md properly)
        st.markdown(
            '<div class="assistant-msg">',
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Tool indicator expander with full debug details
        debug = msg.get("debug")
        if debug and debug.get("agent"):
            summary = get_debug_summary(debug)
            with st.expander(summary, expanded=False):
                render_debug_details(debug)


# ─── Chat Input ──────────────────────────────────────────────────────────────

prompt = None
if st.session_state.get("pending_message"):
    prompt = st.session_state.pending_message
    st.session_state.pending_message = None
elif user_input := st.chat_input("Ask me anything about your accounts..."):
    prompt = user_input

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send to backend
    try:
        result = send_message(prompt)
        response = result["response"]
        debug = result.get("debug", {})

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "debug": debug,
            }
        )
    except httpx.HTTPError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "I'm having trouble connecting to the server. Please try again in a moment.",
                "debug": None,
            }
        )

    st.rerun()
