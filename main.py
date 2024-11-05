import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any

import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from openai import OpenAI
import weaviate
from utils.logging_utils import setup_logging

from chains.transcript_retreival_chain import TranscriptRetrievalChain
from prompt_manager import PromptManager
from utils.parsing_tools import parse_llm_output, clean_html

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

class FeedbackSystem:
    def __init__(self, logger):
        self.logger = logger

    async def collect_feedback(self, user_id: str, query: str, response: str, rating: int, comment: str = None):
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment
        }
        self.logger.info(f"User feedback: {feedback}")

def deserialize_journal_data(filename='./data/journal_data.json'):
    """Deserialize the journal data from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

class TranscriptAgent:
    def __init__(self, cl_instance=None, alpha=.75):
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.weaviate_client = weaviate.connect_to_local()
        self.cl_instance = cl_instance
        self.logger = setup_logging()
        self.logger.info("Transcipt initialized")
        self.feedback_system = FeedbackSystem(self.logger)

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               max_len=50,
                                               return_messages=True)
        
        self.transcript_chain = TranscriptRetrievalChain(
            llm=self.llm,
            weaviate_client=self.weaviate_client,
            openai_client=self.openai_client,
            cl_instance=cl_instance,
            alpha=alpha
        )
        

    async def route_and_execute(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs["question"]
        chat_history = inputs["chat_history"]
        result = await self.transcript_chain.ainvoke({"question": question, "task": "retrieve", "chat_history": chat_history})
        
        print(f"Query: {question}")
        print(f"Response: {result['answer']}")
        
        return result

    async def process_query(self, user_query: str, chat_history: Dict[str, str], user_id: str) -> str:
        print("Incoming query: ", user_query)
        result = await self.route_and_execute({"question": user_query, "chat_history": chat_history})
        answer = result.get("answer", "I'm sorry, I couldn't generate a response for this query.")
        self.memory.chat_memory.add_user_message(user_query)
        self.memory.chat_memory.add_ai_message(answer)
        print(f"User {user_id} - Query: {user_query}")
        print(f"User {user_id} - Response: {answer}")
        return answer


# Chainlit setup
@cl.on_chat_start
async def setup_chain():
    agent = TranscriptAgent(cl_instance=cl)
    cl.user_session.set("agent", agent)
    user_id = str(uuid.uuid4())
    cl.user_session.set("user_id", user_id)
    agent.logger.info(f"New session started for user {user_id}")

    hello = "Welcome to the Podcast Transcript Assistant! 👋🧬🔬"
    welcome_message = """I'm here to help you with various transcript-related tasks:
    How can I assist you with your podcast information today?
    """

    elements = [
        cl.Text(name="Instructions", content=welcome_message, display="inline")
    ]

    message_history = []
    cl.user_session.set("message_history", message_history)

    await cl.Message(content=hello, elements=elements).send()


async def process_message(message: cl.Message, agent: TranscriptAgent, user_id: str):
    message_history = cl.user_session.get("message_history")
    new_user_message = message.content
    response = await agent.process_query(new_user_message, message_history, user_id)
    message_history.append("User: " + message.content)
    message_history.append("AI: " + response) # Add the AI response to the message history
    
    html_content, summary = parse_llm_output(response)
    
    if html_content:
        cleaned_html = clean_html(html_content)
        await cl.Message(content=cleaned_html).send()
    
    if summary:
        await cl.Message(content=summary).send()
    
    # Ask for feedback
    actions = [
        cl.Action(name="rate_response", value="1", label="⭐"),
        cl.Action(name="rate_response", value="2", label="⭐⭐"),
        cl.Action(name="rate_response", value="3", label="⭐⭐⭐"),
        cl.Action(name="rate_response", value="4", label="⭐⭐⭐⭐"),
        cl.Action(name="rate_response", value="5", label="⭐⭐⭐⭐⭐"),
    ]
    await cl.Message(content="How would you rate this response?", actions=actions).send()
    

@cl.action_callback("rate_response")
async def on_action(action: cl.Action):
    rating = int(action.value)
    await cl.Message(content=f"Thank you for rating the response {rating} stars. Would you like to add a comment?").send()
    cl.user_session.set("pending_rating", rating)


@cl.on_message
async def process_feedback(message: cl.Message):
    agent = cl.user_session.get("agent")
    user_id = cl.user_session.get("user_id")
    rating = cl.user_session.get("pending_rating")
    
    if rating is not None:
        last_human_message = agent.memory.chat_memory.messages[-2]
        last_ai_message = agent.memory.chat_memory.messages[-1]
        await agent.feedback_system.collect_feedback(user_id, last_human_message.content, last_ai_message.content, rating, message.content)
        cl.user_session.set("pending_rating", None)
        await cl.Message(content="Thank you for your feedback!").send()
    else:
        # If there's no pending rating, process the message as a normal query
        await process_message(message, agent, user_id)


@cl.on_stop
async def on_stop(self):
    agent = cl.user_session.get("agent")
    agent.memory.chat_memory.clear()
    self.weaviate_client.close()
    agent.logger.info("Session ended")

if __name__ == "__main__":
    import chainlit as cl
    cl.run()