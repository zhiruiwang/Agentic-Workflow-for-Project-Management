# Import the AugmentedPromptAgent class
from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# knowledge source and persona impact
print("\nKnowledge source: The agent uses the LLM's general knowledge (e.g. gpt-3.5-turbo training data) to answer the factual question; no external knowledge document is provided. The system prompt only constrains how the answer is presented, not what facts are used.")
print("Persona impact: The system prompt instructs the agent to assume a college professor persona and to start with 'Dear students,' so the response is formatted as if addressing a class rather than as plain fact.")
