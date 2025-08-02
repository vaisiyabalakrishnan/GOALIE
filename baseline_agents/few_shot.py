from textworld import Agent
from utils.agent_utils import *


class FewShotAgent(Agent):
    def __init__(self, temperature = 0.2):
        self.goal = ""
        self.history = []
        self.task_type = ""
        self.temperature = temperature


    def reset(self, env):
        self.goal = ""
        self.history = []
        env.request_infos.admissible_commands = True


    def act(self, infos, reward, done, obs):
        current_obs_text, admissible_cmds = process_step_inputs(obs, infos)

        if not self.goal:
            self.goal = extract_goal(obs[0]) or "unknown"
            self.task_type = extract_task_type(self.goal)
        
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        prompt = few_shot_prompt(self.goal,self.history,current_obs_text,admissible_cmds)
        print(f"\n[FEW-SHOT PROMPT] {prompt}")

        response, usage = chat_with_llm(prompt)
        print(f"\n[RAW RESPONSE] {response}")

        action, thought = parse_response(response)
        print(f"\n[ACTION BY LLM] {action}")

        for key in total_usage:
                total_usage[key] += usage.get(key, 0)
 
        self.history.append({
            "step": len(self.history) + 1,
            "goal": self.goal,
            "observation": current_obs_text,
            "action": action,
            "done": done
        })

        return action, total_usage
    

def extract_task_type(goal_text):
    # specifying gaol task type 
    goal = goal_text.lower()
    if "look at" in goal or "examine" in goal:
        print(f"[Task Type]: examine in light -> {goal}")
        return "examine_in_light"
    elif "clean" in goal:
        print(f"[Task Type]: clean -> {goal}")
        return "clean_&_place"
    elif "cool" in goal:
        print(f"[Task Type]: cool -> {goal}")
        return "cool_&_place"
    elif "hot" in goal or "heat" in goal:
        print(f"[Task Type]: heat -> {goal}")
        return "heat_&_place"
    elif "put two" in goal or "find two" in goal:
        print(f"[Task Type]: pick two & place -> {goal}")
        return "pick_two_&_place"
    elif "put a" in goal or "put some" in goal:
        print(f"[Task Type]: pick & place -> {goal}")
        return "pick_&_place"
    else:
        return "unknown" 
    

def few_shot_prompt(goal, history, current_obs, admissible):
    return f"""
    You are an intelligent household assistant in a text-based world, you have to compelete a given goal.
    You are to think about your action while refering to examples provided.

    Goal: {goal}


    Examples: {example1},
    {example2}

    History of your actions: 
    {history}

    Current observation:
    {current_obs}

    Admissible actions:
    {admissible}

    Command type:
    {command_types}

    Respond in this format:
    Action: Choose your next move using a valid action or command type. Respond in all lowercase.
    """.strip()


command_types = """
Valid command types follow the structure:
    - look:                             look around your current location
    - inventory:                        check your current inventory
    - go to (receptacle):               move to a receptacle
    - open (receptacle):                open a receptacle
    - close (receptacle):               close a receptacle
    - take (object) from (receptacle):  take an object from a receptacle
    - move (object) to (receptacle):    place an object in or on a receptacle
    - examine (something):              examine a receptacle or an object
    - use (object):                     use an object
    - heat (object) with (receptacle):  heat an object using a receptacle. If no confirmation that action is done in prompt, retry the action.
    - clean (object) with (receptacle): clean an object using a receptacle. If no confirmation that action is done in prompt, retry the action.
    - cool (object) with (receptacle):  cool an object using a receptacle. If no confirmation that action is done in prompt, retry the action.
    - slice (object) with (object):     slice an object using a sharp object. If no confirmation that action is done in prompt, retry the action.
"""


example1 = """
        Goal: put a clean cloth in bathtubbasin

        Observation: You are in the middle of a room. Looking quickly around, you see towelholder 1, toilet 1, bathtubbasin 1, drawer 5, and countertop 1.
        Thought: My goal is to place a clean cloth in the bathtubbasin. First, I need to locate a cloth, pick it up, and find a suitable location to clean it—likely the sinkbasin. Once the cloth is clean, I can place it into the bathtubbasin. A cloth is likely stored in a drawer.
        Action: go to drawer 5

        Observation: You have arrive at drawer 5. The drawer 5 is closed.
        Thought: I need to open drawer 5 to see what's inside.
        Action: open drawer 5

        Observation: You open the drawer 5. The drawer 5 is open. In it, you see a candle 1.
        Thought: There is no cloth in drawer 5. I need to explore another location.
        Action: go to countertop 1
        
        Observation: You arrive at countertop 1. On the countertop 1, you see a cloth 2, a soapbottle 1, a cloth 1, and a candle 3. 
        Thought: There are many items on countertop 1, including two cloth. I should take one and clean it.
        Action: take cloth 1 from countertop 1

        Observation: You pick up the cloth 1 from the countertop 1.
        Thought: Now that I have cloth 1, I should clean it at the sinkbasin.
        Action: go to sinkbasin 1

        Observation: You arrive at sinkbasin 1. On the sinkbasin 1, you see nothing.
        Thought: I need to clean cloth 1 using the sinkbasin 1 to obtain a clean cloth.
        Action: clean cloth 1 with sinkbasin 1

        Observation: You clean the cloth 1 using the sinkbasin 1.
        Thought: Cloth 1 is now clean. To complete the goal, I need to put it into the bathtubbasin.
        Action: go to bathtubbasin 1

        Observation: You arrive at bathtubbasin 1. On the bathtubbasin 1, you see a soapbar 1.
        Thought: I need to put clean cloth into bathtubbasin to complete the goal.
        Action: move cloth 1 to bathtubbasin 1

        Goal completed!
        """
    

example2 = """ 
        Goal: find two remotecontrol and put them in armchair.

        Observation: You are in the middle of a room. Looking quickly around you, you see a dresser 1, a sidetable 2, a sidetable 3, a armchair 1, a sidetable 5 and a sofa 1.
        Thought: My goal is to find two remotecontrol and put them in armchair. First, I need to locate two remotecontrol, and move them to a armchair. Remotecontrols are often place at the sidetable or dresser.
        Action: take remotecontrol 1 from sidetable 2

        Observation:You pick up the remotecontrol 1 from the sidetable 2.
        Thought: I now have one remotecontrol. I should place it in armchair 1, then go look for another remotecontrol.
        Action: go to armchair 1

        Observation: You arrive at armchair 1. On the armchair 1, you see nothing.
        Thought: I need to place remotecontrol 1 on armchair 1.
        Action: move remotecontrol 1 to armchair 1
        
        Observation: You put the remotecontrol 1 in/on the armchair 1.
        Thought: I now need to search for another remotecontrol, since the goal requires two. I'll check the dresser next.
        Action: go to dresser 1

        Observation: You arrive at dresser 1. On the dresser 1, you see a creditcard 1, a television 1, a laptop 1, a book 1, a statue 2, and a cellphone 2.
        Thought: There is no remotecontrol here. I need to search another location.
        Action: go to sidetable 5

        Observation: You arrive at sidetable 5. On the sidetable 5, you see a box 1, and a creditcard 2.
        Thought: Still no remotecontrol found. I need to check elsewhere.
        Action: go to sofa 1

        Observation: You arrive at sofa 1. On the sofa 1, you see a newspaper 1, a pillow 1, and a remotecontrol 2.
        Thought: I've found the second remotecontrol. Now I should pick it up and move it to the same armchair where I placed remotecontrol 1.
        Action: take remotecontrol 2 from sofa 1

        Observation: You pick up the remotecontrol 2 from the sofa 1.
        Thought: I need to go to armchair 1 and place this second remotecontrol there.
        Action: go to armchair 1

        Observation: You arrive at armchair 1. On the armchair 1, you see remotecontrol 1.
        Thought: I need to place remotecontrol 2 on armchair 1.
        Action: move remotecontrol 2 to armchair 1

        Goal completed!
        """