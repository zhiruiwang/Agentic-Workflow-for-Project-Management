# Import the KnowledgeAugmentedPromptAgent class from workflow_agents
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
# Instantiate a KnowledgeAugmentedPromptAgent with the given persona and knowledge
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

response = knowledge_agent.respond(prompt)
# Print statement demonstrating the agent uses the provided knowledge (London) rather than inherent LLM knowledge (Paris)
print(response)
print("\n(Expected: The agent should answer 'London' per the provided knowledge, not 'Paris' from its training.)")
