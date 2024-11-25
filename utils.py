import streamlit as st

def validate_prompt(prompt: str) -> bool:
    """
    Validate the input prompt
    """
    if not prompt or len(prompt.strip()) < 3:
        return False
    return True

def setup_page():
    """
    Configure the Streamlit page settings
    """
    st.set_page_config(
        page_title="Pixel Art Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            margin-top: 1rem;
        }
        .stTextArea > div > div > textarea {
            height: 100px;
        }
        </style>
    """, unsafe_allow_html=True)
