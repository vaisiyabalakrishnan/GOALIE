from textworld import Agent
import time
from utils.agent_utils import *


class ReActAgent(Agent):
    def __init__(self, keep_history=True, temperature=0.2):
        self.keep_history = keep_history
        self.temperature = temperature
        self.history = []
        self.goal = ""


    def reset(self, env):
        env.request_infos.admissible_commands = True
        self.history = []
        self.goal = ""


    def act(self, infos, reward, done, obs):
        if not self.goal:
            self.goal = extract_goal(obs[0]) or "unknown"
        
        current_obs_text, cmds = process_step_inputs(obs, infos)

        # Build ReAct prompt
        print("[REACT DEBUG] Building ReAct prompt...")
        prompt = build_react_prompt(self.goal, self.history[-5:], obs[:1000], cmds)  # Truncate history + observation to 1000 characters
        print("[REACT DEBUG] Prompt content:\n", prompt)

        # Send prompt to LLM
        start = time.time()
        print("[REACT DEBUG] Sending to LLM...")
        response_text, usage = chat_with_llm(prompt)
        end = time.time()
        print(f"[REACT DEBUG] LLM responded in {end - start:.2f} seconds.")
        print(f"[REACT DEBUG] Raw LLM Response:\n{response_text}\n")
        print(f"[ReAct DEBUG] Token usage: {usage}\n")

        # Get action and thought from LLM response
        action, thought = parse_response(response_text)
        print(f"[REACT DEBUG] Chosen action: {action}")
        print(f"[REACT DEBUG] Thought: {thought}")
            
        # Add step to history
        if self.keep_history:
            self.history.append({
            'step': len(self.history) + 1,
            'observation': current_obs_text,
            'thought': thought,
            'action': action,
            'done?': done })
            
        return action, usage
    

def build_react_prompt(goal, history, observation, admissible):
    # thought_chain = "\n".join(history)
    command_types = """
    Valid command types follow the structure:
        look:                             look around your current location
        inventory:                        check your current inventory
        go to (receptacle):               move to a receptacle
        open (receptacle):                open a receptacle
        close (receptacle):               close a receptacle
        take (object) from (receptacle):  take an object from a receptacle
        move (object) to (receptacle):    place an object in or on a receptacle
        examine (something):              examine a receptacle or an object
        use (object):                     use an object
        heat (object) with (receptacle):  heat an object using a receptacle. Ensure object is being carried. If no confirmation in prompt, retry the action.
        clean (object) with (receptacle): clean an object using a receptacle. Ensure object is being carried. If no confirmation in prompt, retry the action.
        cool (object) with (receptacle):  cool an object using a receptacle. Ensure object is being carried. If no confirmation in prompt, retry the action.
        slice (object) with (object):     slice an object using a sharp object. Ensure object is being carried. If no confirmation in prompt, retry the action.
    """
    
    return f"""
    You are an intelligent household assistant in a text-based world.

    Goal: {goal}

    Here is the current observation:
    {observation}

    Valid command types: {command_types}

    Valid actions: {', '.join(admissible)}

    History of your thoughts and actions:
    {history}

    You should think step-by-step to reason about the environment and then decide the next action. Use this format:

    Thought: ...
    Action: <choose one from valid actions>

    Only output the next Thought and Action. Be concise.
    """.strip()