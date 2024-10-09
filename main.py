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
        self.weaviate_client = weaviate.Client("http://localhost:8080")
        self.cl_instance = cl_instance
        # self.logger = setup_logging()
        # self.logger.info("Transcipt initialized")

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        self.transcript_chain = TranscriptRetrievalChain(
            llm=self.llm,
            weaviate_client=self.weaviate_client,
            openai_client=self.openai_client,
            cl_instance=cl_instance,
            alpha=alpha
        )
        

    async def route_and_execute(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs["question"]
        result = await self.transcript_chain.ainvoke({"question": question, "task": "retrieve"})
        
        print(f"Query: {question}")
        print(f"Response: {result['answer']}")
        
        return result

    async def process_query(self, query: str, user_id: str) -> str:
        self.memory.chat_memory.add_user_message(query)
        result = await self.route_and_execute({"question": query})
        answer = result.get("answer", "I'm sorry, I couldn't generate a response for this query.")
        print(f"User {user_id} - Query: {query}")
        print(f"User {user_id} - Response: {answer}")
        return answer

# Chainlit setup
@cl.on_chat_start
async def setup_chain():
    agent = TranscriptAgent(cl_instance=cl)
    cl.user_session.set("agent", agent)
    user_id = str(uuid.uuid4())
    cl.user_session.set("user_id", user_id)
    # agent.logger.info(f"New session started for user {user_id}")

    hello = "Welcome to the Podcast Transcript Assistant! 👋🧬🔬"
    welcome_message = """
    I'm here to help you with various transcript-related tasks:

    Important: Our interactions are stateless. Please provide all necessary context in each query.

    How can I assist you with your podcast information today?
    """

    elements = [
        cl.Text(name="Instructions", content=welcome_message, display="inline")
    ]

    await cl.Message(content=hello, elements=elements).send()

async def process_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    user_id = cl.user_session.get("user_id")
    response = await agent.process_query(message.content, user_id)
    
    html_content, summary = parse_llm_output(response)
    
    if html_content:
        cleaned_html = clean_html(html_content)
        await cl.Message(content=cleaned_html).send()
    
    if summary:
        await cl.Message(content=summary).send()
    

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
        await process_message(message)

if __name__ == "__main__":
    import chainlit as cl
    cl.run()