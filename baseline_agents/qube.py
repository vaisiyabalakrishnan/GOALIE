from textworld import Agent
import time
from utils.agent_utils import chat_with_llm, parse_response, process_step_inputs
from react import ReActAgent
from utils.derailment_detector import DerailmentDetector


class QuBEAgent(ReActAgent):
    def __init__(self, keep_history=True, temperature=0.2):
        super().__init__(keep_history, temperature)
        self.partial_qube_agent = PartialQubeAgent(keep_history, temperature)
        self.detector = DerailmentDetector(max_repeats=3)
    
    def reset(self, env):
        super().reset(env)
        self.partial_qube_agent.reset(env)

    def act(self, infos, reward, done, obs):
        # Store the last observation for derailment detection
        self.detector.last_obs = obs[0]  
        
        # If derailed, invoke QuBE
        if self.detector.is_derailed():
            print("\n [QUBE DEBUG] Detected derailment. Invoking QuBE reasoning.\n")
            action, thought, usage = self.partial_qube_agent.act(infos, reward, done, obs, self.history, self.goal)

            if self.keep_history:
                self.history.append({
                'step': len(self.history) + 1,
                'observation': obs[0],
                'thought': thought,
                'action': action,
                'done?': done })
        
        # If not, simply use ReAct
        else:
            action, usage = super().act(infos, reward, done, obs)
        
        # Update derailment detector
        self.detector.update(action)

        return action, usage
    



class PartialQubeAgent(Agent):
    def __init__(self, keep_history=True, temperature=0.2):
        self.keep_history = keep_history
        self.temperature = temperature
        # NOTE: this is Qube-specific history
        self.history = []
        self.goal = ""
        self.last_action_str = ""

    def reset(self, env):
        env.request_infos.admissible_commands = True
        self.history = []
        self.goal = ""
        self.last_action_str = ""

    def act(self, infos, reward, done, obs, ReActHistory, goal):
        if not self.goal:
            self.goal = goal

        current_obs_text, cmds = process_step_inputs(obs, infos)
        
        task_status = "Completed" if done else "Incomplete"

        rationale = ""
        belief_state = ""

        belief_prompt = belief_state_prompt(
            format_history_for_prompt(ReActHistory), current_obs_text
        )
        belief_state = chat_with_llm(belief_prompt)
        if isinstance(belief_state, tuple):
            belief_state = belief_state[0]
        print(f"[QUBE DEBUG] Belief State:\n{str(belief_state).strip()}\n")

        rationale_prompt = rationale_generation_prompt(
            "FAILED_ACTION", format_history_for_prompt(ReActHistory), belief_state
        )
        rationale = chat_with_llm(rationale_prompt)
        if isinstance(rationale, tuple):
            rationale = rationale[0]
        print(f"[QUBE DEBUG] Rationale:\n{str(rationale).strip()}\n")

        # Final Qube prompt
        prompt = build_qube_prompt(
            current_obs_text,
            self.goal,
            cmds,
            format_history_for_prompt(ReActHistory),
            rationale,
            task_status,
        )

        # Send prompt to LLM
        start = time.time()
        
        print(f"[QUBE DEBUG] Sending Qube prompt to LLM...\n{prompt}\n")
        response_text, usage = chat_with_llm(prompt)
        end = time.time()
        print(f"[QUBE DEBUG] LLM responded in {end - start:.2f} seconds.")
        print(f"[QUBE DEBUG] Raw LLM Response:\n{response_text}\n")

        # Get action and thought from LLM response
        action, thought = parse_response(response_text)
        print(f"[QUBE DEBUG] Chosen Action: {action}")
        print(f"[QUBE DEBUG] Thought: {thought}")

        # Add step to QuBe's history
        if self.keep_history:
            self.history.append(
                {
                    # EDITED: recording step number based on ReAct history length
                    "step": len(ReActHistory) + 1,
                    "observation": current_obs_text,
                    "thought": thought,
                    "action": action,
                    "rationale": rationale,
                    "belief_state": belief_state,
                    "done?": done,
                }
            )

        # Store last action string for future use
        self.last_action_str = action
        
        return action, thought, usage
    

def format_history_for_prompt(history):
    history_str = ""
    for i in range (len(history)):
        history_str += f"Step {i+1}.Observation: {history[i]['observation']}\n"
        history_str += f"Step {i+1}.Thought: {history[i]['thought']}\n"
        history_str += f"Step {i+1}.Action: {history[i]['action']}\n"
    return history_str.strip()


def belief_state_prompt(history, obs):
    return f"""
    You are an agent acting in a text-based household environment, operating by following goals at any given time.

    As a world model, your job is to accurately provide information about the current environment.
    Here is your current trajectory: {history.strip()}.

    Current Observation: {obs}

    Answer the following questions. Here are the questions:
    1) Where am I now?
    2) What is my inventory?
    3) Which receptacles are available?
    4) Which receptacles do not need to be checked again?
    For each question, repeat the question itself, then answer it.
    If you do not know an answer, answer I don't know.
    """.strip()


def rationale_generation_prompt(action, history, belief_state):
    return f"""
    You are an agent acting in a text-based household environment, operating by following goals at any given time.
    You are solving a task in a text-based household environment.

    This is what you know so far: {belief_state}
    Your trajectory originally was {history[-1]}, but it led to a failure. 
    Write a new thought that is more likely to lead to a successful trajectory, and only base your answer on what you know.
    """.strip()


def build_qube_prompt(obs, goal, admissible_actions, history, rationale, task_status):
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
    You are an intelligent agent navigating a text-based household environment to achieve a specific goal.
    You operate by thinking step-by-step, taking an action, and then observing the outcome.

    Goal: 
    {goal}

    Task Completion Status: 
    {task_status}

    Rationale from your Belief State:
    {rationale.strip()}

    Current Observation:
    {obs}

    Valid command types: {command_types}
    Available specific actions: {', '.join(admissible_actions)}

    History:
    {history.strip()}

    Follow this format:
    Thought: Your rationale based on the belief state. Think step-by-step about what to do next.
    Action: Choose a command from the 'Available specific actions'. Do not come up with command on your own.

    """.strip()