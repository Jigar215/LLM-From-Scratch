import streamlit as st
import time
from inference import generate_response

# --- UI CONFIGURATION ---
st.set_page_config(page_title="LawGPT", page_icon="⚖️", layout="centered")
st.title("⚖️ LawGPT")
st.caption("Indian Legal Assistant powered by a custom Transformer built from scratch.")

# --- INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LawGPT. Ask me a question about the Indian Penal Code or Constitution."}
    ]

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- HANDLE USER INPUT ---
if prompt := st.chat_input("E.g., What is the punishment for murder?"):
    
    # 1. Display user's question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Generate model's response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Indian Law texts..."):
            
            # Call your custom inference engine directly (it already returns a string)
            raw_output = generate_response(prompt)
            
            # --- POST-PROCESSING FILTER (The Garbage Collector) ---
            # Isolate the response from the prompt
            try:
                answer = raw_output.split("### Response:\n")[-1].strip()
            except Exception:
                answer = raw_output 
            
            # Clean up hallucinated tags
            if answer.startswith("### Answer:"):
                answer = answer.replace("### Answer:", "", 1).strip()
            elif answer.startswith("### Response:"):
                answer = answer.replace("### Response:", "", 1).strip()
                
            # Cut off run-on instructions
            if "### Instruction:" in answer:
                answer = answer.split("### Instruction:")[0].strip()
                
            # Remove the end token
            answer = answer.replace("<|endoftext|>", "").strip()
            # ------------------------------------------------------
            
            # Typewriter effect for a professional feel
            placeholder = st.empty()
            full_response = ""
            for char in answer:
                full_response += char
                placeholder.markdown(full_response + "▌")
                time.sleep(0.01) # adjust speed if needed
            placeholder.markdown(full_response)
            
    # 3. Save to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})