import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Page config ──
st.set_page_config(
    page_title="HTML Code Generator",
    page_icon="💻",
    layout="wide"
)

st.title("💻 HTML Code Generator")
st.caption("Powered by Fine-tuned Llama 3.2 | jinesh90/llama-3.2-ft-html-generator")

# ── Load model (cached) ──
@st.cache_resource
def load_model():
    base_model = "unsloth/Llama-3.2-1B-Instruct"
    adapter    = "jinesh90/llama-3.2-ft-html-generator"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype = torch.float32,  # CPU needs float32
        device_map  = "cpu",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer

with st.spinner("Loading model... (first time takes ~2 min)"):
    model, tokenizer = load_model()

st.success("Model loaded!")

# ── Build prompt ──
def build_prompt(instruction, input_text=""):
    return f"""Generate valid HTML code based on the following instructions. \
If additional input is provided, update the provided code accordingly. \
Return only the HTML code without explanations or formatting.

### Instruction
{instruction}

### Input
{input_text if input_text else "(none)"}"""

# ── Generate HTML ──
def generate(instruction, input_text=""):
    messages = [{"role": "user", "content": build_prompt(instruction, input_text)}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
        )
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# ── UI Layout ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input")
    instruction = st.text_area(
        "Instruction",
        placeholder="Create an HTML login form with email and password fields",
        height=100
    )
    input_code = st.text_area(
        "Input code (optional)",
        placeholder="Paste existing HTML here if you want to modify it...",
        height=100
    )
    generate_btn = st.button("🚀 Generate HTML", type="primary")

with col2:
    st.subheader("✨ Generated Output")
    if generate_btn and instruction:
        with st.spinner("Generating HTML..."):
            result = generate(instruction, input_code)
        st.code(result, language="html")
        st.subheader("🌐 Live Preview")
        st.components.v1.html(result, height=300, scrolling=True)
    elif generate_btn and not instruction:
        st.warning("Please enter an instruction!")
    else:
        st.info("Enter an instruction and click Generate!")
