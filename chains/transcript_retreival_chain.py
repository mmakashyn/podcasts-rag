import asyncio
import json
from typing import Dict, Any, List, Optional
from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from prompt_manager import PromptManager
from utils.parsing_tools import format_retreival_query


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

    def sort_by_timestamp(self, results: List[Dict]) -> List[Dict]:
        def timestamp_to_seconds(timestamp: str) -> int:
            minutes, seconds = map(int, timestamp.split(':'))
            return minutes * 60 + seconds
        return sorted(results, key=lambda transcript: timestamp_to_seconds(transcript['timestamp']))

    def _retrieve_transcripts(self, query: str, n: int = 20, rich_author_meta=False) -> List[Dict]:

        query_formatted = format_retreival_query(self.llm, query)

        print("Formatted query: ", query_formatted)
        
        #self.cl_instance
        

        query_embedding = self.openai_client.embeddings.create(input=[query_formatted], model='text-embedding-3-large').data[0].embedding

        fields = ["timestamp", "text"]
    
        results = (
            self.weaviate_client.query
            .get("PodcastChunk", fields)
            .with_hybrid(
                query=query,
                vector=query_embedding,
                alpha=self.alpha
            )
            .with_limit(n)
            .with_additional(["score"])
            .do()
        )

        return results['data']['Get']['PodcastChunk']

    def transcript_synthesize(self, results: List[Dict], original_question: str, task: str) -> str:
        system_prompt = PromptManager.get_transcript_retrieval_prompt()
        
        transcripts = results

        # Format the transcripts for the prompt
        formatted_transcripts = json.dumps(transcripts, indent=2)
        human_message_template = """Original question: {question}
        PodcastChunk:
        {transcripts}
        Please {task} the information from these transcripts to answer the original question."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_message_template),
        ])

        # import pdb; pdb.set_trace()

        response = self.llm(prompt.format_messages(
            question=original_question,
            transcripts=formatted_transcripts,
            task = task,
        ))

        return response.content

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:
        question = inputs["question"]
        task = inputs.get("task", "summary")  # Default to summary if task is not provided
        
        # Retrieve transcripts
        results = await asyncio.to_thread(self._retrieve_transcripts, question)
        
        results_filtered = await asyncio.to_thread(self.sort_by_timestamp, results)
       
        # Synthesize final answer
        final_answer = await asyncio.to_thread(self.transcript_synthesize, results_filtered, question, task)
        
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
        results_filtered = self.sort_by_timestamp(results)

        # Synthesize final answer
        final_answer = self.transcript_synthesize(results_filtered, question, task)
        
        return {"answer": final_answer}