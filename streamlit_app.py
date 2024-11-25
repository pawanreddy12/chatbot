import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch

# Title and description
st.title("💬 Chatbot Using Fine-Tuned T5")
st.write(
    "This chatbot uses a fine-tuned T5 model hosted on Hugging Face to generate responses. "
    "You can interact with the chatbot below."
)

@st.cache_resource
def load_model():
    try:
        model_name = "pawanreddy/peft_t5_fine_tuned_model"
        st.write(f"Loading model from Hugging Face Hub: {model_name}")
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the base T5 model
        base_model_name = "google/t5-base-lm-adapt"  # Base model used for fine-tuning
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Load PEFT adapter configuration
        peft_config = PeftConfig.from_pretrained(model_name)
        
        # Apply the PEFT adapter to the base model
        model = PeftModel.from_pretrained(base_model, model_name)
        
        st.write("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise

tokenizer, model = load_model()

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field
if user_input := st.chat_input("Type your message here..."):

    # Display the user's input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a response from the fine-tuned model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Move model and inputs to the appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = tokenizer(
                f"Generate the content of the article titled '{user_input}'",
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(device)
            model.to(device)

            # Generate response using Seq2Seq-specific generation
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.9,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if not response.strip() or "?" in response:
                response = "I'm sorry, I didn't understand that. Can you please rephrase?"

        # Display the assistant's response
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
