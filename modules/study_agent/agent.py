"""
The Study Planner Agent — the brain of EduMentor AI.

This is a ReAct-style agent (Reasoning + Acting):
1. It receives a goal and student context
2. It THINKS about what to do
3. It ACTS by calling a tool
4. It OBSERVES the result
5. It repeats until the goal is achieved

This is what makes it "agentic" — it autonomously decides
which tools to use and in what order, rather than following
a fixed script.
"""

import json
from groq import Groq
from config.settings import GROQ_API_KEY
from modules.study_agent.tools import (
    tool_assess_student,
    tool_create_study_plan,
    tool_generate_resources,
    tool_check_progress,
)

_client = Groq(api_key=GROQ_API_KEY)

# Tool registry — maps tool names to actual functions
TOOLS = {
    "assess_student":    tool_assess_student,
    "create_study_plan": tool_create_study_plan,
    "generate_resources": tool_generate_resources,
    "check_progress":    tool_check_progress,
}

TOOL_DESCRIPTIONS = """
Available tools:
1. assess_student(student_profile) — Analyze student strengths/weaknesses
2. create_study_plan(student_profile, available_hours_per_day, exam_days_away, priority_topics) — Build personalized schedule
3. generate_resources(topic, topic_text) — Create summary, flashcards, quiz for a topic
4. check_progress(session_results) — Evaluate if student is improving
"""


class StudyPlannerAgent:
    """
    Autonomous agent that creates and manages personalized study plans.
    Uses ReAct pattern: Think → Act → Observe → Repeat.
    """

    def __init__(self, max_steps: int = 6):
        self.max_steps   = max_steps
        self.memory      = []    # stores the full agent trajectory
        self.tool_results = {}   # stores results from each tool call

    def run(self, goal: str, context: dict) -> dict:
        """
        Run the agent to achieve a goal given student context.

        Args:
            goal:    what we want the agent to accomplish
                     e.g. "Create a 5-day study plan for this student"
            context: dict with student_profile, topic_texts, etc.

        Returns:
            dict with the final plan, resources, and agent reasoning trace
        """
        print(f"\nAgent starting: {goal}\n")
        self.memory      = []
        self.tool_results = {}

        system_prompt = f"""You are an intelligent Study Planner Agent for EduMentor AI.
Your goal: {goal}

{TOOL_DESCRIPTIONS}

You work in steps. Each step, you must respond with ONLY this JSON:
{{
  "thought": "Your reasoning about what to do next",
  "action": "tool_name",
  "action_input": {{...tool arguments...}},
  "done": false
}}

When you have enough information to complete the goal, respond with:
{{
  "thought": "I have everything I need",
  "action": "finish",
  "final_answer": "Summary of what was accomplished",
  "done": true
}}

Student Context:
{json.dumps(context.get("student_profile", {}), indent=2)}

Available topics: {list(context.get("topic_texts", {}).keys())}
"""

        messages = [{"role": "system", "content": system_prompt}]

        for step in range(self.max_steps):
            print(f"Step {step + 1}/{self.max_steps}")

            # Add previous observations to messages
            if self.memory:
                last_obs = f"Previous tool result: {json.dumps(self.memory[-1].get('observation', {}))[:500]}"
                messages.append({"role": "user", "content": last_obs})

            # Agent thinks and decides what to do
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )

            raw    = response.choices[0].message.content.strip()
            raw    = raw.replace("```json", "").replace("```", "").strip()

            try:
                decision = json.loads(raw)
            except Exception:
                print(f"  Could not parse agent response: {raw[:100]}")
                break

            thought = decision.get("thought", "")
            action  = decision.get("action", "")
            print(f"  Thought: {thought[:100]}")
            print(f"  Action: {action}")

            # If agent is done, collect final answer
            if decision.get("done") or action == "finish":
                print("\nAgent completed task.")
                self.memory.append({
                    "step":        step + 1,
                    "thought":     thought,
                    "action":      "finish",
                    "observation": decision.get("final_answer", ""),
                })
                break

            # Execute the chosen tool
            observation = self._execute_tool(action, decision.get("action_input", {}), context)
            self.tool_results[action] = observation

            self.memory.append({
                "step":        step + 1,
                "thought":     thought,
                "action":      action,
                "action_input": decision.get("action_input", {}),
                "observation":  observation,
            })

            messages.append({
                "role":    "assistant",
                "content": raw,
            })

        return {
            "goal":         goal,
            "steps_taken":  len(self.memory),
            "tool_results": self.tool_results,
            "reasoning":    self.memory,
        }

    def _execute_tool(self, tool_name: str, tool_input: dict, context: dict) -> dict:
        """Execute a tool and return its result."""
        if tool_name not in TOOLS:
            return {"error": f"Unknown tool: {tool_name}"}

        print(f"  Executing: {tool_name}")

        try:
            if tool_name == "assess_student":
                return TOOLS[tool_name](context["student_profile"])

            elif tool_name == "create_study_plan":
                return TOOLS[tool_name](
                    student_profile          = context["student_profile"],
                    available_hours_per_day  = tool_input.get("available_hours_per_day", 2),
                    exam_days_away           = tool_input.get("exam_days_away", 7),
                    priority_topics          = tool_input.get("priority_topics",
                                               context["student_profile"].get("weak_topics", [])[:4])
                )

            elif tool_name == "generate_resources":
                topic      = tool_input.get("topic", "")
                topic_text = context.get("topic_texts", {}).get(topic, f"Study material for {topic}")
                return TOOLS[tool_name](topic, topic_text)

            elif tool_name == "check_progress":
                return TOOLS[tool_name](context.get("session_results", []))

            else:
                return TOOLS[tool_name](**tool_input)

        except Exception as e:
            return {"error": str(e), "tool": tool_name}