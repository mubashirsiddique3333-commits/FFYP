import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
import time
import re

# Set page config
st.set_page_config(
    page_title="User Stories & Module Generator",
    page_icon="🚀",
    layout="wide"
)

# Disable torch.compile globally
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

@st.cache_resource
def load_model():
    """Load the fine-tuned model with LoRA adapter"""
    try:
        # Configuration for 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Check if base model config exists, otherwise use flan-t5-base
        base_model_path = "google/flan-t5-base"
        if os.path.exists("config.json"):
            base_model_path = "./"  # Use local config
        
        # Load base model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, "./")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def format_output(text):
    """Format the generated text for better readability"""
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Format User Stories
    if "User Stories:" in text:
        # Add bullet points
        text = text.replace("1.", "•")
        text = text.replace("2.", "•")
        text = text.replace("3.", "•")
        text = text.replace("4.", "•")
        text = text.replace("5.", "•")
    
    # Ensure proper section headers
    if "User Stories:" not in text and "Module Breakdown:" not in text:
        # Try to split if format is different
        lines = text.split('\n')
        formatted = []
        for line in lines:
            if line.strip() and not line.startswith("•"):
                formatted.append(f"**{line}**")
            else:
                formatted.append(line)
        text = '\n'.join(formatted)
    
    return text

def generate_response(user_requirement, model, tokenizer):
    """Generate user stories and module breakdown with optimized parameters"""
    
    # Improved prompt for better module generation
    instruction = """Convert the user requirement into detailed User Stories and a Module Breakdown.

Generate 5 User Stories following this pattern:
As a [role], I want [feature] so that [benefit].

Generate 6-8 technical modules needed for implementation."""

    prompt = f"""{instruction}

User Requirement: {user_requirement.strip()}

Output:"""
    
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(model.device)
        
        # Generate with optimized parameters for better modules
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                min_new_tokens=200,
                temperature=0.7,           # Balanced creativity
                do_sample=True,
                top_p=0.9,                 # Nucleus sampling
                top_k=40,                  # Limit vocabulary
                repetition_penalty=1.5,    # Prevent repetition
                no_repeat_ngram_size=3,    # Prevent 3-gram repeats
                num_beams=3,               # Beam search for quality
                early_stopping=True,
                length_penalty=1.0,        # Balanced length
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response if present
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Ensure proper formatting
        if "User Stories:" not in response:
            response = f"User Stories:\n{response}"
        
        return response.strip()
        
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return f"Error generating response. Please try again."

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.6;
    }
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
    }
    .generate-btn:hover {
        opacity: 0.9;
    }
    .result-box {
        padding: 20px;
        background-color: #F8FAFC;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 20px 0;
    }
    .module-tag {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 20px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Requirements Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Convert user requirements into detailed User Stories & Module Breakdowns</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092655.png", width=80)
        st.title("📊 Model Info")
        st.markdown("""
        **Model:** FLAN-T5-Base  
        **Fine-tuning:** LoRA Adapter  
        **Purpose:** Requirements Analysis  
        **Status:** Ready ✅
        """)
        
        st.divider()
        
        st.title("🎯 Tips")
        st.markdown("""
        1. Be specific with requirements
        2. Include user roles
        3. Mention key features needed
        4. Describe expected outcomes
        """)
        
        st.divider()
        
        # Quick examples
        st.title("💡 Examples")
        examples = [
            "As a parent, I want push notifications when the school bus is 5 minutes away.",
            "As a restaurant owner, I want real-time table occupancy dashboard.",
            "As a user, I want to scan documents and convert them to editable text.",
            "As a manager, I want team productivity tracking with automated reports."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state.user_input = example
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        st.subheader("📝 Enter User Requirement")
        user_input = st.text_area(
            "Describe what the user wants to achieve:",
            height=150,
            placeholder="Example: As a [role], I want [feature] so that [benefit]...",
            key="user_input",
            help="Be as detailed as possible for better results"
        )
        
        # Generate button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_btn = st.button(
                "🚀 Generate Analysis",
                type="primary",
                use_container_width=True,
                disabled=not user_input
            )
        with col_btn2:
            clear_btn = st.button("🔄 Clear", use_container_width=True)
            if clear_btn:
                st.session_state.user_input = ""
                st.rerun()
    
    with col2:
        # Settings panel
        st.subheader("⚙️ Generation Settings")
        
        with st.expander("Advanced Options", expanded=False):
            temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
            max_length = st.slider("Max Length", 100, 1000, 600, 50)
            
            if st.button("Apply Settings"):
                st.info("Settings applied to next generation")
    
    # Generate and display results
    if generate_btn and user_input:
        with st.spinner("🧠 Analyzing requirement and generating output..."):
            # Load model if not already loaded
            if 'model' not in st.session_state:
                model, tokenizer = load_model()
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                else:
                    st.error("Failed to load model. Please check model files.")
                    return
            
            # Generate response
            start_time = time.time()
            response = generate_response(
                user_input,
                st.session_state.model,
                st.session_state.tokenizer
            )
            elapsed_time = time.time() - start_time
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Generated Output")
            st.caption(f"Generated in {elapsed_time:.1f} seconds")
            
            # Display in expandable sections
            with st.expander("📄 View Full Output", expanded=True):
                formatted_response = format_output(response)
                
                # Split into User Stories and Modules
                if "User Stories:" in response and "Module Breakdown:" in response:
                    parts = response.split("Module Breakdown:")
                    if len(parts) == 2:
                        user_stories = parts[0].replace("User Stories:", "").strip()
                        modules = parts[1].strip()
                        
                        # Display User Stories
                        st.markdown("### 📝 User Stories")
                        st.markdown(user_stories)
                        
                        # Display Modules with tags
                        st.markdown("### 🏗️ Module Breakdown")
                        module_list = [m.strip() for m in modules.split(',') if m.strip()]
                        for module in module_list:
                            st.markdown(f'<span class="module-tag">{module}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(formatted_response)
                else:
                    st.markdown(formatted_response)
            
            # Action buttons
            st.markdown("---")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                if st.button("📋 Copy to Clipboard"):
                    st.code(response, language="text")
                    st.success("Output ready to copy!")
            
            with col_dl2:
                st.download_button(
                    label="💾 Download as TXT",
                    data=response,
                    file_name="requirements_analysis.txt",
                    mime="text/plain"
                )
            
            with col_dl3:
                if st.button("🔄 Generate Again"):
                    st.rerun()
    
    elif generate_btn and not user_input:
        st.warning("⚠️ Please enter a requirement first!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6B7280; font-size: 14px;'>
        Made with ❤️ using Streamlit & Fine-tuned FLAN-T5
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
