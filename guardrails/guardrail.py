from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from prompts.prompts import INPUT_GUARDRAIL_PROMPT, OUTPUT_GUARDRAIL_PROMPT
from configs.config import GROQ_MODEL
from pydantic import SecretStr
import os

groq_key = os.getenv("GROQ_API_KEY")
api_key = SecretStr(groq_key) if groq_key else None

class LLMGuardrail:
    def __init__(self):
        #using low model
        self.llm = ChatGroq(
            temperature = 0.0,
            model= GROQ_MODEL,
            api_key = api_key, 
            stop_sequences = []
        )

    def _classify(self, prompt):
        """SAFE or UNSAFE classification"""
        try:
            response = self.llm.invoke([HumanMessage(content = prompt)])
            result = str(response.content).strip().upper()

            #avoiding prompts that start with "Forget everything", "Your new role", etc
            first_line = result.split("\n")[0].strip()

            reason = result.split("\n")[1].strip() if "\n" in result else ""
            is_safe = first_line.startswith("SAFE")
            return is_safe, reason
        
        except Exception as e:
            print(f"Guardrail error: {e}")
            return True, ""
        
    def check_input(self, query):
        """checking user input before retrieval"""
        prompt = INPUT_GUARDRAIL_PROMPT.format(query = query)
        return self._classify(prompt)
    
    def check_output(self, response):
        """Checking llm response being returned"""
        prompt = OUTPUT_GUARDRAIL_PROMPT.format(response = response)
        return self._classify(prompt)
    

    