import streamlit as st
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)

# Title
st.title('LLAMA Index Demo')
st.divider()

# Load Tokenizer and Model
st.title('Model name and auth token')
model_name = st.text_input('Enter your Hugging Face model name', value="meta-llama/Llama-2-7b-chat-hf")
auth_token = st.text_input('Enter your Hugging Face auth token', value="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")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Tokenizer and Model
@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./model/', token=auth_token)
    # Create model
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    )
    model = AutoModelForCausalLM.from_pretrained(model_name,
            cache_dir='./model/', token=auth_token,
            quantization_config=quantization_config,
            rope_scaling={"type":"dynamic", "factor":2},
            max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
    )
    return tokenizer, model

tokenizer, model = get_tokenizer_model()
print("Model loaded")

# prompt ="### User:How many majors in SEOULTECH?### Assistant:"
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input('User: ')

if prompt:
    # update chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Here... text streamer does not work as well as I intended with streamlit
# I will try to fix this later
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        # model inference
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)
        output = model.generate(**inputs, streamer=streamer,
                        use_cache=True, max_new_tokens=100)
        output_text = tokenizer.decode(output[0],skip_special_tokens=True)
        
        placeholder = st.empty()
        full_response = ""
        for item in output_text:
            full_response += item
            placeholder.markdown(full_response)
        placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": output_text})

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.button('Clear Chat History', on_click=clear_chat_history)
