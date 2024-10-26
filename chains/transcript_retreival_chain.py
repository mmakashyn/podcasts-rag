import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from weaviate.classes.query import MetadataQuery
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompt_manager import PromptManager
from utils.parsing_tools import format_retreival_query
from itertools import groupby
from operator import itemgetter


class TranscriptRetrievalChain(Chain):
    llm: BaseLanguageModel
    weaviate_client: Any
    openai_client: Any
    cl_instance: Any
    alpha: float = 0.75
    
    class Config:
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def sort_by_timestamp_and_podcast(self, results: List[Dict]) -> List[Dict]:
        # Define a helper function to parse the timestamp
        def parse_timestamp(item):
            # Extract the timestamp string
            timestamp_str = item.properties.get('timestamp', '00:00:00.000')
            try:
                # Try parsing the timestamp with milliseconds
                time_obj = datetime.strptime(timestamp_str, "%H:%M:%S.%f")
            except ValueError:
                # If it fails (i.e., no milliseconds), try without milliseconds
                time_obj = datetime.strptime(timestamp_str, "%H:%M:%S")
            return time_obj
    
        # Sort the list of results based on the parsed timestamp
        sorted_results = sorted(results, key=parse_timestamp)

        # Add the "score" to the "properties" and return the list
        for item in sorted_results:
            item.properties['score'] = item.metadata.score
    
        results = [item.properties for item in sorted_results]
        # Group by podcast_name
        grouped_results = []
        for podcast_name, group in groupby(results, key=itemgetter('podcast_name')):
            grouped_results.extend(list(group))
        
        return grouped_results

    def _retrieve_transcripts(self, query: str, n: int = 20, rich_author_meta=False) -> List[Dict]:
        query_formatted = format_retreival_query(self.llm, query)

        print("Formatted query: ", query_formatted)
        
        query_embedding = self.openai_client.embeddings.create(input=[query_formatted], model='text-embedding-3-large').data[0].embedding
        results = (
            self.weaviate_client.collections.get("PodcastChunk").query
            .hybrid(
                query=query,  # Replace with relevant search term
                vector=query_embedding,
                alpha=0.75,  # Adjust this value to balance between vector and keyword search
                limit=n,
                return_metadata=MetadataQuery(score=True)
            )
        )
        return results.objects

    def transcript_synthesize(self, results: List[Dict], original_question: str, chat_history: Dict[str, str], task: str) -> str:
        system_prompt = PromptManager.get_transcript_retrieval_prompt()
        
        transcripts = results

        # Format the transcripts for the prompt
        formatted_transcripts = json.dumps(transcripts, indent=2)
        human_message_template = """Original question: {question}
        Previous chat history:
        {chat_history}
        PodcastChunk:
        {transcripts}
        Please {task} the information from these transcripts to answer the original question and take into account the chat history if needed."""
        
        prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", human_message_template)
                ]
        )

        # import pdb; pdb.set_trace()

        final_prompt = prompt.format_messages(
            question=original_question,
            transcripts=formatted_transcripts,
            task = task,
            chat_history=chat_history
        )

        response = self.llm(final_prompt)

        return response.content

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:
        question = inputs["question"]
        chat_history = inputs["chat_history"]
        task = inputs.get("task", "summary")  # Default to summary if task is not provided
        
        # Retrieve transcripts
        results = await asyncio.to_thread(self._retrieve_transcripts, question)
        
        results_filtered = await asyncio.to_thread(self.sort_by_timestamp_and_podcast, results)
       
        # Synthesize final answer
        final_answer = await asyncio.to_thread(self.transcript_synthesize, results_filtered, question, chat_history, task)
        
        return {"answer": final_answer}

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:
        question = inputs["question"]
        task = inputs.get("task", "summary")  # Default to summary if task is not provided

        # Retrieve articles
        results = self._retrieve_transcripts(question)
        results_filtered = self.sort_by_timestamp_and_podcast(results)

        # Synthesize final answer
        final_answer = self.transcript_synthesize(results_filtered, question, task)
        
        return {"answer": final_answer}