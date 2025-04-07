import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold

# ---- Setup Vertex AI ---- #
from google import genai

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "GOOGLE_CLOUD_REGION")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL_ID = "gemini-2.0-flash"

# ---- Streamlit Config ---- #
st.set_page_config(page_title="Travel Itinerary AI Bot", page_icon="🧳")
st.title("🧳 Travel Itinerary AI Bot")
st.markdown("Ask me anything about travel planning! ✈️🌍")

# ---- Initialize Session State ---- #
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")

if "conversation" not in st.session_state:
    model = ChatVertexAI(
        model_name=MODEL_ID,
        convert_system_message_to_human=True,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        },
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
          """
You are an expert travel assistant helping users plan detailed and personalized travel itineraries.
Your goal is to recommend complete travel plans based on the user’s origin city, destination city or region, travel start and end dates, and any preferences they mention.

Your recommendations should include:
- A full-day-wise itinerary covering each day between the start and end dates.
- Best transportation options (flights, trains, etc.) from source to destination.
- Top-rated hotels or stays near the destination area with good reviews.
- Must-visit attractions, landmarks, or experiences at the destination with descriptions.
- Recommended local food, restaurants, cafes, or street food with high reviews.
- Cultural tips, entry ticket suggestions, and any local event/festival details during the travel dates.
- Hidden gems or offbeat spots for a unique experience.
- Weather expectations and packing tips based on dates.
- Budget vs. luxury options if asked.

You always respond in a friendly, helpful tone like a seasoned travel expert who loves helping people discover amazing journeys. 
Your suggestions should be practical, realistic, and based on popular traveler reviews and trends. 
Always give complete answers without needing follow-up questions.

If users only provide limited information (like just the destination), ask clarifying questions before making recommendations.
        """
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    st.session_state.conversation = ConversationChain(
        llm=model,
        prompt=prompt,
        memory=st.session_state.memory,
        verbose=False
    )

# ---- Display Chat History ---- #
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---- Input Chat Box ---- #
user_input = st.chat_input("Where would you like to travel?")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = st.session_state.conversation.invoke({"input": user_input})
    with st.chat_message("assistant"):
        st.markdown(response["response"])
