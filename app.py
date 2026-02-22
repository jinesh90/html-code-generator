import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="HTML Code Generator", page_icon="💻", layout="wide")
st.title("💻 HTML Code Generator")
st.caption("Powered by Fine-tuned Llama 3.2")

# ── Load client (lightweight - no model downloaded!) ──
@st.cache_resource
def load_client():
    token = st.secrets["HUGGINGFACE_TOKEN"]
    return InferenceClient(
        model="jinesh90//llama-3.2-html-generator-merged",
        token=token
    )

client = load_client()

def build_prompt(instruction, input_text=""):
    return f"""Generate valid HTML code based on the following instructions. If additional input is provided, update the provided code accordingly. Return only the HTML code without explanations or formatting.

### Instruction
{instruction}

### Input
{input_text if input_text else "(none)"}"""

def generate(instruction, input_text=""):
    messages = [{"role": "user", "content": build_prompt(instruction, input_text)}]
    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content

# ── UI ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input")
    instruction = st.text_area("Instruction",
        placeholder="Create an HTML login form with email and password",
        height=100)
    input_code = st.text_area("Input code (optional)",
        placeholder="Paste existing HTML to modify...",
        height=100)
    btn = st.button("🚀 Generate HTML", type="primary")

with col2:
    st.subheader("✨ Output")
    if btn and instruction:
        with st.spinner("Generating..."):
            result = generate(instruction, input_code)
        st.code(result, language="html")
        st.subheader("🌐 Live Preview")
        st.components.v1.html(result, height=300, scrolling=True)
    elif btn:
        st.warning("Please enter an instruction!")
    else:
        st.info("Enter an instruction and click Generate!")


