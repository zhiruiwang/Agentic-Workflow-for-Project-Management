# agentic_workflow.py

# Allow importing workflow_agents from phase_1 when running from phase_2
import os
import sys
_phase1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase_1")
if _phase1_dir not in sys.path:
    sys.path.insert(0, _phase1_dir)

# Import the following agents from the workflow_agents.base_agents module
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)
from dotenv import load_dotenv

load_dotenv()

# Load the OpenAI key into a variable called openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
_spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Product-Spec-Email-Router.txt")
with open(_spec_path, "r", encoding="utf-8") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    + product_spec
)
# Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_product_manager, knowledge_product_manager
)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_pm = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_pm,
    product_manager_knowledge_agent,
    max_interactions=10,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_program_manager, knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_pgm = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_pgm,
    program_manager_knowledge_agent,
    max_interactions=10,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_dev_engineer, knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_de = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_de,
    development_engineer_knowledge_agent,
    max_interactions=10,
)


# Job function persona support functions: take a step, call knowledge agent, evaluate, return final response
def product_manager_support_function(input_query):
    result = product_manager_evaluation_agent.evaluate(input_query)
    return result.get("final_response", "")


def program_manager_support_function(input_query):
    result = program_manager_evaluation_agent.evaluate(input_query)
    return result.get("final_response", "")


def development_engineer_support_function(input_query):
    result = development_engineer_evaluation_agent.evaluate(input_query)
    return result.get("final_response", "")


# Routing Agent - list of routes for Product Manager, Program Manager, and Development Engineer
routing_agent = RoutingAgent(openai_api_key, [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories.",
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features by grouping user stories. Does not define user stories or development tasks.",
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining development tasks and engineering work for each user story. Does not define stories or features.",
        "func": development_engineer_support_function,
    },
])

# Run the workflow when script is executed (not when imported)
if __name__ == "__main__":
    print("\n*** Workflow execution started ***\n")
    # Workflow Prompt
    workflow_prompt = "What would the development tasks for this product be?"
    print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

    print("\nDefining workflow steps from the workflow prompt")
    # Implement the workflow
    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
    completed_steps = []

    for i, step in enumerate(workflow_steps, 1):
        print(f"\n--- Step {i}: {step} ---")
        result = routing_agent.route(step)
        completed_steps.append(result)
        print(f"Result:\n{result}")

    print("\n*** Workflow execution completed ***")

    # Produce a final, structured output for the Email Router project
    if completed_steps and workflow_steps:
        print("\n" + "=" * 60)
        print("FINAL, STRUCTURED OUTPUT: Email Router Project Plan")
        print("(Comprehensively planned project per product specification)")
        print("=" * 60)
        for i, (step_name, step_output) in enumerate(zip(workflow_steps, completed_steps), 1):
            print(f"\n--- {i}. {step_name} ---\n")
            print(step_output)
        print("\n" + "=" * 60)
        print("END OF EMAIL ROUTER PROJECT PLAN")
        print("=" * 60)
    elif completed_steps:
        print("\n--- Final output of the workflow (last completed step) ---")
        print(completed_steps[-1])