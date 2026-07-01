import streamlit as st
import requests
import json

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #0F111A;
    }
    .recommendation-card {
        border-radius: 12px;
        padding: 16px;
        background-color: #1E2235;
        border-left: 5px solid #00F2FE;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-title {
        font-size: 1.15rem;
        font-weight: bold;
        color: #00F2FE;
        margin-bottom: 4px;
    }
    .recommendation-link {
        font-size: 0.85rem;
        color: #4facfe;
        text-decoration: none;
    }
    .recommendation-type {
        display: inline-block;
        font-size: 0.75rem;
        background-color: #313852;
        color: #E2E8F0;
        padding: 2px 8px;
        border-radius: 4px;
        margin-top: 6px;
        font-weight: bold;
    }
    .turn-counter {
        font-size: 0.95rem;
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 8px;
        background-color: #1E2235;
        color: #E2E8F0;
        border: 1px solid #2D3748;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/chat"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "end_of_conversation" not in st.session_state:
    st.session_state.end_of_conversation = False

with st.sidebar:
    st.title("🤖 SHL AI Assistant")
    st.write("Find the perfect SHL assessments for your hiring needs.")
    
    user_turns = 0
    for m in st.session_state.messages:
        if m["role"] == "user":
            user_turns += 1
            
    st.markdown(f'<div class="turn-counter">Turns used: <b>{user_turns} / 8</b></div>', unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("Reset Conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.recommendations = []
        st.session_state.end_of_conversation = False
        st.rerun()
        
    st.write("---")
    st.markdown("""
    ### ⚙️ Guidelines
    * Be specific about the **role**, **seniority**, and **skills**.
    * Ask to **compare** tests if you need to choose.
    * Let the agent know when you are **satisfied** with the shortlist.
    """)

st.title("💬 Conversational Assessment Recommender")

col_chat, col_recs = st.columns([2, 1])

with col_recs:
    st.subheader("📋 Shortlisted Assessments")
    if st.session_state.recommendations:
        for rec in st.session_state.recommendations:
            type_mapping = {
                "K": "Knowledge & Skills (K)",
                "P": "Personality & Behavior (P)",
                "S": "Simulations (S)",
                "A": "Ability & Aptitude (A)",
                "B": "Biodata & Situational Judgment (B)",
                "C": "Competencies (C)",
                "D": "Development & 360 (D)",
                "E": "Assessment Exercises (E)",
                "U": "Unknown (U)"
            }
            rec_type = type_mapping.get(rec["test_type"], rec["test_type"])
            
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-title">{rec['name']}</div>
                <a class="recommendation-link" href="{rec['url']}" target="_blank">View Assessment Details ↗</a><br>
                <div class="recommendation-type">Type: {rec_type}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No assessments recommended yet. Describe your requirements to the chat assistant to get recommendations!")

with col_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.end_of_conversation:
        st.success("🎉 Conversation completed successfully! The final recommended assessments list is displayed on the sidebar.")
    elif user_turns >= 8:
        st.warning("⚠️ Turn limit reached (Max 8 turns). Reset the conversation to start over.")
    else:
        if prompt := st.chat_input("What profile are you hiring for?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            payload_messages = []
            for m in st.session_state.messages:
                payload_messages.append({
                    "role": m["role"],
                    "content": m["content"]
                })
            payload = {"messages": payload_messages}
            
            with st.spinner("Analyzing requirements & fetching recommendations..."):
                try:
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        reply = data.get("reply", "")
                        recommendations = data.get("recommendations", [])
                        end_of_conv = data.get("end_of_conversation", False)
                        
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        with st.chat_message("assistant"):
                            st.markdown(reply)
                            
                        if recommendations:
                            current_names = set()
                            for r in st.session_state.recommendations:
                                current_names.add(r["name"])
                            for r in recommendations:
                                if r["name"] not in current_names:
                                    st.session_state.recommendations.append(r)
                                    
                        st.session_state.end_of_conversation = end_of_conv
                        st.rerun()
                    else:
                        st.error(f"Error from server (Status Code: {response.status_code})")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI server. Ensure it is running at {API_URL}. Details: {str(e)}")
