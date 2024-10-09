from prompts.system_prompts import *

class PromptManager:

    @staticmethod
    def get_transcript_retrieval_prompt():
        return TRANSCRIPT_RETRIEVAL_SYSTEM_PROMPT
    
    @staticmethod
    def get_retreival_format_prompt():
        return RETREIVAL_FORMAT_PROMPT