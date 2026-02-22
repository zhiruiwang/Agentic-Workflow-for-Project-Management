#!/usr/bin/env python3
"""
Run all seven phase_1 agent test scripts and capture terminal output to text files.

Outputs are saved in the run_outputs/ folder. Each file shows the printed prompts
and agent responses for rubric submission (screenshots or text capture of terminal output).

Test scripts: DirectPromptAgent, AugmentedPromptAgent, KnowledgeAugmentedPromptAgent,
RAGKnowledgePromptAgent, EvaluationAgent, RoutingAgent, ActionPlanningAgent.
"""

import subprocess
import sys
from pathlib import Path

# All seven test scripts required by the rubric
TEST_SCRIPTS = [
    "direct_prompt_agent.py",
    "augmented_prompt_agent.py",
    "knowledge_augmented_prompt_agent.py",
    "rag_knowledge_prompt_agent.py",
    "evaluation_agent.py",
    "routing_agent.py",
    "action_planning_agent.py",
]

# Output folder (created next to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "run_outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Running all phase_1 test scripts. Outputs will be saved to: {OUTPUT_DIR}\n")

    for script in TEST_SCRIPTS:
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            print(f"  SKIP (not found): {script}")
            continue

        out_name = script.replace(".py", "_output.txt")
        out_path = OUTPUT_DIR / out_name
        print(f"  Running: {script} -> {out_name}")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(SCRIPT_DIR),
                capture_output=True,
                text=True,
                timeout=300,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"=== stdout ===\n{result.stdout}")
                if result.stderr:
                    f.write(f"\n=== stderr ===\n{result.stderr}")
                if result.returncode != 0:
                    f.write(f"\n=== exit code: {result.returncode} ===\n")
            if result.returncode != 0:
                print(f"    Warning: exit code {result.returncode}")
            else:
                print(f"    Done.")
        except subprocess.TimeoutExpired:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("(Run timed out after 300 seconds.)\n")
            print(f"    Timeout.")
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"(Error running script: {e})\n")
            print(f"    Error: {e}")

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
