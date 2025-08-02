from textworld import Agent
from utils.derailment_detector import *
from utils.agent_utils import *
from goalie_utils import *
from goalie_prompt import*


class Goalie(Agent):
    def __init__(self, temperature=0.2):
        self.goal = ""
        self.task_type = ""
        self.history = []
        self.location = []        
        self.inventory = []
        self.main_plan = []
        self.use_few_shot = False
        self.temperature = temperature
        self.latest_summary = "Nothing yet."
        self.derailment_detector = DerailmentDetector()


    def reset(self, env):
        self.goal = ""
        self.task_type = ""
        self.main_plan = []
        self.history.clear()
        self.location.clear()
        self.inventory.clear()
        self.use_few_shot = False
        self.latest_summary = "Nothing yet."
        self.derailment_detector = DerailmentDetector()


    def act(self, infos, reward, done, obs):
        current_obs_text, admissible_cmds = process_step_inputs(obs, infos)
        print(f"\n[Observation from ALFWorld] {current_obs_text}")
        
        picked_item = extract_picked_object(current_obs_text)
        if picked_item and picked_item not in self.inventory:
            self.inventory.append(picked_item)

        moved_item = extract_moved_object(current_obs_text)
        if moved_item and moved_item in self.inventory:
            self.inventory.remove(moved_item)

        inventory_str = ", ".join(self.inventory) if self.inventory else "Nothing yet."
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if not self.goal:
            self.goal = extract_goal(obs[0]) or "unknown"
            self.task_type = extract_task_type(self.goal)
            self.location = extract_locations(current_obs_text)

            raw_main_plan, usage = chat_with_llm(decomposer_prompt(self.goal, self.task_type))
            self.main_plan = parse_main_plan(raw_main_plan)
            print(f"\n[LLM Main Plan] {self.main_plan}")
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

        if self.history:
            self.derailment_detector.update(self.history[-1]["action"])
            self.derailment_detector.last_obs = current_obs_text
            self.use_few_shot = self.derailment_detector.is_derailed()

        if self.history:
            self.derailment_detector.update(self.history[-1]["action"])
            self.derailment_detector.last_obs = current_obs_text
            self.use_few_shot = self.derailment_detector.is_derailed()

        if self.use_few_shot:
            print("\n[DERAILMENT DETECTED] Triggering few-shot fallback...\n")
            prompt = build_derailed_prompt(self.goal, self.latest_summary, current_obs_text, admissible_cmds, self.task_type)
        else:
            prompt = executor(self.main_plan, self.latest_summary, inventory_str, current_obs_text, self.location)

        raw_reply, usage = chat_with_llm(prompt)
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)
        
        thought, action, summary = parse_executor(raw_reply)
        self.latest_summary = summary
        print(f"\n[LLM Thought] Thought: {thought}")
        print(f"\n[LLM Action] Action: {action}")
        print(f"\n[LLM Summary] Summary: {self.latest_summary}\n")
    
        self.history.append({
            "step": len(self.history) + 1,
            "observation": current_obs_text,
            "summary": self.latest_summary,
            "thought": thought,
            "action": action,
            "Derailment": self.use_few_shot,
            "done": done
        })

        return action, total_usage, self.main_plan