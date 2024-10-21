import streamlit as st
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_random
import json

# Inicializa el historial en la sesión si aún no existe
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

def log_retry_attempt(retry_state):
    attempt = retry_state.attempt_number
    st.warning(f"Attempt {attempt} failed for {retry_state.args[0]}. Retrying...")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random(min=60, max=120),
    before_sleep=log_retry_attempt,
    retry_error_callback=lambda retry_state: (
        retry_state.args[0],
        "Error: All retry attempts failed",
    ),
)
async def get_llm_response(llm, query):
    url = "http://localhost:8001/get_response"
    headers = {"Content-Type": "application/json"}
    data = {"llm": llm, "query_str": query}

    async with httpx.AsyncClient() as client:
        try:
            post_response = await client.post(
                url, headers=headers, json=data, timeout=600
            )
            post_response.raise_for_status()
            result = post_response.json()
            
            return llm, result["response"], result['nodes']
        except httpx.TimeoutException:
            return llm, "Error: Request timed out after 10 minutes"
        except httpx.HTTPStatusError as e:
            return llm, f"Error: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return llm, f"Error: {str(e)}"

async def get_all_responses(llms, query):
    tasks = [get_llm_response(llm, query) for llm in llms]
    return await asyncio.gather(*tasks)

# llm_options = ["openai"]
# selected_llms = st.multiselect("Select LLM(s):", llm_options)

st.write("### Conversation History:")
for message in st.session_state['message_history']:
    role = message["role"]
    content = message["message"]
    if role == "user":
        st.chat_message("user").write(content)
    elif role == 'assistant':
        st.chat_message("assistant").markdown(content)

if prompt := st.chat_input(
    "What was BNP Paribas Group's net income attributable to equity holders "
    "for the first half of 2024, and how does it compare to the same period in 2023?"
):
    st.session_state['message_history'].append({"role": "user", "message": prompt})
    
    st.chat_message("user").write(prompt)

    responses = asyncio.run(get_all_responses(['openai'], prompt))

    for llm, response, sources in responses:
        if not sources:
            st.chat_message("assistant").markdown(f"{response}")
            
            st.session_state['message_history'].append({
                "role": "assistant",
                "message": f"{response}"
            })
        else:
            assistant_response = f"{response}\n\n"

            assistant_response += "<h3>Sources:</h3>\n"
            for source in sources:
                filename = source.get("filename")
                content = source.get("content")
                
                assistant_response += f'<p><strong>File:</strong> <code>{filename}</code></p>'
                assistant_response += f'<div style="white-space: pre-wrap; background-color: rgba(240,240,240,0.03); padding: 10px; border-radius: 5px;">{content}</div><br>'

            st.chat_message("assistant").markdown(assistant_response, unsafe_allow_html=True)

            # Añadimos la respuesta y los sources al historial
            st.session_state['message_history'].append({
                "role": "assistant",
                "message": assistant_response
            })
