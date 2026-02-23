# Import the OpenAI class from the openai library
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime

# DirectPromptAgent class definition
class DirectPromptAgent:
    """Agent that relays user prompts directly to the LLM without additional context."""

    def __init__(self, openai_api_key):
        """Initialize the agent with the OpenAI API key."""
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """Generate a response using the OpenAI API."""
        print("\n--- DirectPromptAgent: Prompt ---")
        print(prompt)
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        print("\n--- DirectPromptAgent: Response ---")
        return response.choices[0].message.content
        
# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """Agent that responds according to a predefined persona via a system prompt."""

    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        system_content = f"{self.persona} Forget all previous context."
        print("\n--- AugmentedPromptAgent: System Prompt ---")
        print(system_content)
        print("\n--- AugmentedPromptAgent: User Prompt ---")
        print(input_text)
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        print("\n--- AugmentedPromptAgent: Response ---")
        return response.choices[0].message.content

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """Agent that incorporates specific provided knowledge and a persona when responding."""

    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        system_content = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            "Answer the prompt based on this knowledge, not your own."
        )
        print("\n--- KnowledgeAugmentedPromptAgent: System Prompt ---")
        print(system_content)
        print("\n--- KnowledgeAugmentedPromptAgent: User Prompt ---")
        print(input_text)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        print("\n--- KnowledgeAugmentedPromptAgent: Response ---")
        return response.choices[0].message.content

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            single_chunk = [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
            with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
                writer.writeheader()
                writer.writerow({"text": text, "chunk_size": len(text)})
            return single_chunk

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            chunk_id += 1
            if end >= len(text):
                break
            # Always advance start so the loop terminates (avoids infinite loop when overlap >= chunk length).
            start = max(start + 1, end - self.chunk_overlap)

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.
        Streams row-by-row to avoid loading all embeddings into memory (safe for constrained environments).

        Returns:
        None. Use find_prompt_in_knowledge() to query; embeddings are stored on disk.
        """
        chunks_path = f"chunks-{self.unique_filename}"
        embeddings_path = f"embeddings-{self.unique_filename}"
        with open(chunks_path, "r", encoding="utf-8", newline="") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = list(reader.fieldnames or ["text", "chunk_size"]) + ["embeddings"]
            with open(embeddings_path, "w", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    text = row.get("text", "")
                    emb = self.get_embedding(text)
                    row["embeddings"] = str(emb)
                    writer.writerow(row)
        return None

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.
        Reads the embeddings CSV row-by-row (no pandas) to minimize memory use in constrained environments.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        best_sim = -1.0
        best_chunk = ""
        embeddings_path = f"embeddings-{self.unique_filename}"
        with open(embeddings_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emb = np.array(eval(row["embeddings"]))
                sim = self.calculate_similarity(prompt_embedding, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_chunk = row["text"]

        system_content = f"You are {self.persona}, a knowledge-based assistant. Forget previous context."
        user_content = f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"
        print("\n--- RAGKnowledgePromptAgent: System Prompt ---")
        print(system_content)
        print("\n--- RAGKnowledgePromptAgent: User Prompt ---")
        print(user_content)
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0
        )
        print("\n--- RAGKnowledgePromptAgent: Response ---")
        return response.choices[0].message.content

class EvaluationAgent:
    """Agent that evaluates worker agent responses against criteria and refines through feedback."""

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        """Initialize the EvaluationAgent with given attributes."""
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """Manage interactions between worker and evaluator to achieve a solution meeting criteria."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        response_from_worker = None
        evaluation = None
        iterations = 0

        for i in range(self.max_interactions):
            iterations = i + 1
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            print("\n--- EvaluationAgent: Evaluator System Prompt ---")
            print(self.persona)
            print("\n--- EvaluationAgent: Evaluator User Prompt ---")
            print(eval_prompt)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{self.persona}"},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                print("\n--- EvaluationAgent: Instruction Prompt ---")
                print(instruction_prompt)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )

        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": iterations
        }

class RoutingAgent:
    """Agent that routes user prompts to the most appropriate specialized agent by semantic similarity."""

    def __init__(self, openai_api_key, agents):
        """Initialize the agent with given attributes."""
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        """Calculate the embedding of the text using the text-embedding-3-large model."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def route(self, user_input):
        """Route user prompt to the agent with the highest similarity to the prompt."""
        print("\n--- RoutingAgent: User Input ---")
        print(user_input)
        input_emb = self.get_embedding(user_input)
        input_emb = np.array(input_emb)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue
            agent_emb = np.array(agent_emb)
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

class ActionPlanningAgent:
    """Agent that extracts ordered steps from a user prompt using provided knowledge."""

    def __init__(self, openai_api_key, knowledge):
        """Initialize the agent attributes."""
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        """Extract and return a list of action steps from the user prompt."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from the user prompt "
            "the steps requested to complete the action the user is asking for. You return the steps as a list. "
            "Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )
        print("\n--- ActionPlanningAgent: System Prompt ---")
        print(system_prompt)
        print("\n--- ActionPlanningAgent: User Prompt ---")
        print(prompt)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        response_text = response.choices[0].message.content or ""
        print("\n--- ActionPlanningAgent: Response ---")

        steps = [s.strip() for s in response_text.split("\n") if s.strip()]
        return steps