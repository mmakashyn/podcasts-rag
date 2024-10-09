import re
from bs4 import BeautifulSoup, Comment
from datetime import datetime
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate 
from prompt_manager import PromptManager
from typing import Dict, Any, List, Optional, Tuple
import json


def parse_llm_output(response: str):
    """
    Parse the LLM output  n into HTML and summary parts.
    """
    # Split the response into HTML and summary parts
    parts = re.split(r'(</html>)', response, maxsplit=1)
    
    if len(parts) > 1:
        html_content = parts[0] + parts[1]
        summary = parts[2].strip() if len(parts) > 2 else ""
    else:
        html_content = ""
        summary = response.strip()
    
    return html_content, summary


def clean_html(html_content: str):
    """
    Clean and format the HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Remove comments
    comments = soup.findAll(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    
    # Ensure the table has border and cellpadding
    table = soup.find('table')
    if table:
        table['border'] = '1'
        table['cellpadding'] = '5'
    
    return str(soup)


def format_retreival_query(llm: BaseLanguageModel, user_query: str) -> str:

    human_message = f"Format this user query for articles retreival: {user_query}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", PromptManager.get_retreival_format_prompt()),
        ("human", human_message),
    ])

    response = llm(prompt.format_messages())
    # import pdb; pdb.set_trace()
    
    return response.content.strip()