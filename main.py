import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
def call_llm(question):
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash-lite',
        temperature=0,
        top_p=1,
        top_k=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=GEMINI_API_KEY
    )
    return llm.invoke(question).content

def respond_to_chat(message, history):
    # Extract the current question
    if isinstance(message, dict):
        question = message.get('text', '')
    else:
        question = str(message)
    
    if not question:
        return "No question provided!"
    
    try:
        context_messages = []
        recent_history = history[-5:] if history else []
        
        for exchange in recent_history:
            if isinstance(exchange, dict):
                if exchange.get('role') == 'user':
                    context_messages.append(f"Human: {exchange.get('content', '')}")
                elif exchange.get('role') == 'assistant':
                    context_messages.append(f"Assistant: {exchange.get('content', '')}")
            elif isinstance(exchange, (list, tuple)) and len(exchange) >= 2:
                user_msg, assistant_msg = exchange[0], exchange[1]
                if user_msg:
                    context_messages.append(f"Human: {user_msg}")
                if assistant_msg:
                    context_messages.append(f"Assistant: {assistant_msg}")
        
        # Build the full prompt with context
        if context_messages:
            context_str = "\n".join(context_messages)
            full_prompt = f"Previous conversation:\n{context_str}\n\nCurrent question: {question}"
        else:
            full_prompt = question
        
        return call_llm(full_prompt)
        
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages", height=800)

    chat = gr.ChatInterface(
        chatbot=chatbot,
        fn=respond_to_chat
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)