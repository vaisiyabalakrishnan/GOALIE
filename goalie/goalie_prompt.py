from goalie_utils import *


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


def decomposer_prompt(goal_text, task_type):
    return f"""
    Your task is to decompose the given goal into subgoals that are straightforward and executable by an LLM agent operating in ALFWorld.

    Further instruction for {task_type} task: {task_type_instuction(task_type)}

    Goal: {goal_text}

    Instructions:
    - Break the goal into a numbered list of concise subgoals forming a coherent plan. Do NOT assume the location of objects or the location of target receptacles.
    - Think step-by-step how to complete the goal. Try to make it an action per step only.
    - Do NOT include actions like "close microwave" or similar unless explicitly stated.
    - Keep each step short, direct, and logically grounded.   

    Return your output like this:
    Main Plan:
    1. First subgoal. [Thought]: <Your reasoning why>
    2. Next subgoal. [Thought]: <Your reasoning why>
    3. ...
    ...
    """.strip()     


def executor(main_plan_steps, summary, inventory, current_obs, extracted_location):
    return f"""
    You are playing an intelligent household assistant agent designed to solve problems in a text-based world using the Reasoning and Acting approach.
    
    Your task is to exercute based on this plan given to you make sure to follow the steps: 
    {main_plan_steps}

    What you have: 
    {inventory}

    Current observation: 
    {current_obs}

    Go through this list of locations FULLY and figure out which location might contain object needed or location not visited: 
    {extracted_location}
    
    Command type Valid:
    {command_types}

    Previous summary of location explore and action taken: 
    {summary}

    Think step by step with reasoning.

    Respond in this format for one step. 
    Thought: Reason step-by-step about current location, current observation, what the overall goal is, and what you need to do next to make progress towards the goal. Think of household environment, consider which location and action to take in this step and why.
    Action: Craft your next move using "Command type Valid". Respond in all lowercase.

    Summary: Can you summarize all location you've explore, action taken and keep all previous summary? Include action performed by agent, location agent visit, and objects only from "observation" at that location. 
    Current location: TO Update what is your current location?

    Locations visited: [Create a list of location visited]

    (Only if there's object at location)(Objects at [Location]: [List of Objects Only from "observation" in history]), ...

    some previous action taken:
    [Action] ([Thought in history (summarize a short version. Keep it short)])
    ...
    """.strip()

def build_derailed_prompt(goal, summary, observation, admissible, task_type):
    return f"""
    You are an intelligent household assistant agent designed to solve problems in a text-based world using the Reasoning and Acting approach.
    You seem to be stuck—repeating actions and not making progress toward your goal. It's time to re-evaluate and adapt your strategy.

    Your goal: {goal}

    Task type: {task_type}

    Current observation: {observation}

    Valid command types: {command_types}

    Summary of all actions taken so far: {summary}

    Examples of successful strategies: {few_shot_examples(task_type)}

    Guidance for Recovery:
    - Re-evaluate: What have you tried? What were the results? Identify specific missteps or dead ends.
    - Check the location where you found the object again.
    - Identify Gaps: What information or items do you lack to proceed? Check your inventory if unsure.
    - Learn from Examples: Compare your current situation and past actions to the successful strategies shown above. How did those examples break out of similar impasses?
    - Explore Alternatives: Consider exploring new locations, trying different actions on familiar objects, or restarting a specific subgoal if it's truly unrecoverable.
    - Think Critically: Don't just pick an action. Explain why you are choosing it based on your re-evaluation and the guidance.

    Respond in this format:
    Thought: Reflect on why you're stuck, what new insights you've gained from the examples/guidance, and explain how you will adjust your strategy to make progress.

    Action: Choose your next move using a valid action or valid command type. Respond in all lowercase.

    Summary: Can you summarize all location you've explore, action taken and keep all previous summary? Include action performed by agent, location agent visit, and objects only from "observation" at that location. 
    Current location: TO Update what is your current location?

    Locations visited: [Create a list of location visited]

    (Only if there's object at location)(Objects at [Location]: [List of Objects Only from "observation" in history]), ...

    some previous action taken:
    [Action] ([Thought in history (summarize a short version. Keep it short)])
    ...
    """.strip()


def few_shot_examples(task_type):
    if task_type == "examine_in_light":
        return"""
        Goal: examine the alarmclock with the desklamp.

        Observation: You are in the middle of a room. Looking quickly around you, you see a safe 1, a shelf 4, a drawer 2, a bed 1, a drawer 1, a shelf 5, a shelf 2, a sidetable 2, a shelf 3, a drawer 3, a shelf 1, a sidetable 1, a desk 1, and a garbagecan 1.
        Thought: I need to find the alarmclock first. Pillow would most likely be on a desk. I should go to desk 1 to locate the alarmclock.
        Action: go to desk 1

        Observation: You arrive at desk 1. On the desk 1, you see a pen 1, a bowl 1, a alarmclock 2, a pencil 2, a pencil 3, a creditcard 3, a book 1, a alarmclock 3, a keychain 3, and a book 2.
        Thought: I've found two alarmclock on the desk. I need to pick one up.
        Action: take alarmclock 2 from desk 1
        
        Observation: You pick up the alarmclock 2 from the desk 1.
        Thought: Now I have alarmclock 2 in my inventory. The next step is to find the desklamp to examine it. Desklamp are most liekly on the sidetable.
        Action: go to sidetable 1

        Observation: You arrive at sidetable 1. On sidetable 1, there is a desklamp 1.
        Thought: I need to activate the desklamp in order to example the alarmclock.
        Action: use desklamp 1

        goal completed!
        """
    
    elif task_type == "clean_&_place":
        return """
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

    elif task_type == "cool_&_place":
        return""" 
        Goal: cool a bottle and put it in countertop

        Observation: You are in the middle of a room. Looking quickly around, you see fridge 1, countertop 1, shelf 1, sinkbasin 1, and table 1.
        Thought: My goal is to cool a bottle and place it on the countertop. To do that, I first need to find a bottle. Bottles are likely stored on countertops or on the shelfs. Once I retrieve the bottle, I will use the fridge to cool it before placing it on the countertop.
        Action: go to shelf 1

        Observation: You arrive at shelf 1. On the shelf 1, you see nothing.
        Thought: There is no bottle on the shelf I should check another location.
        Action: go to countertop 1

        Observation: You arrive at countertop 1. On it, you see a bottle 1 and a plate 1.
        Thought: I have found a bottle. I should pick it up and cool it.
        Action: take bottle 1 from countertop 1

        Observation: You pick up bottle 1 from countertop 1.
        Thought: Now that I have the bottle, I need to cool it using a suitable receptacle—likely the fridge.
        Action: go to fridge 1

        Observation: You arrive at fridge 1. The fridge is closed.
        Thought: I need to open fridge 1 to use it for cooling.
        Action: open fridge 1

        Observation: You open fridge 1. Inside, you see milk 1 and juice 2.
        Thought: I should cool bottle 1 using fridge 1.
        Action: cool bottle 1 with fridge 1

        Observation: You cool bottle 1 using fridge 1.
        Thought: Bottle 1 is now cooled. To complete the goal, I need to place it on the countertop.
        Action: go to countertop 1

        Observation: You arrive at countertop 1. On it, you see nothing.
        Thought: Time to move the cooled bottle onto the countertop.
        Action: move bottle 1 to countertop 1

        Goal completed!
        """

    elif task_type == "heat_&_place":
        return"""
        Goal: put a hot potato on countertop

        Observation: You are in the middle of a room. You see a fridge 1, shelf 1, sinkbasin 1, countertop 1, and a microwave 1.
        Thought: My goal is to put a hot potato on the countertop. First, I need to locate a potato it's likely inside the fridge. Then I need to heat the potato using a microwave, and finally place it on the countertop.
        Action: go to fridge 1

        Observation: You arrive at fridge 1. The fridge is closed.
        Thought: To check for a potato, I need to open the fridge.
        Action: open fridge 1

        Observation: You open fridge 1. Inside, you see potato 1 and tomato 1.
        Thought: Great, I've found the potato. I should take it from the fridge.
        Action: take potato 1 from fridge 1

        Observation: You pick up potato 1 from fridge 1.
        Thought: Now I have potato 1 in my inventory. The next step is heat the potato. I should go to the microwave to heat the potato.
        Action: go to microwave 1

        Observation: You arrive at microwave 1. The microwave is closed.
        Thought: I need to open the microwave to heat potato 1.
        Action: open microwave 1

        Observation: The microwave is now open. Inside, you see nothing.
        Thought: Now that microwave is open I need to call the action heat potato 1 with microwave 1 to get a heated potato. There is no need to move or put the object into the microwave.
        Action: heat potato 1 with microwave 1

        Observation: Potato 1 has been heated.
        Thought: Perfect, I heat the potato as observed. The final step is to place it on the countertop.
        Action: go to countertop 1

        Observation: You arrive at countertop 1. On it, you see nothing.
        Thought: I will move the hot potato onto the countertop to complete the goal.
        Action: move potato 1 to countertop 1

        Goal completed!
        """
    
    elif task_type == "pick_two_&_place":
        return""" 
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
    
    elif task_type == "pick_&_place":
        return"""
        Goal: put some watch on shelf.

        Observation: You are in the middle of a room. Looking quickly around you, you see a armchair 2, a armchair 1, a coffeetable 2, a coffeetable 1, a garbagecan 1, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sofa 1, and a tvstand 1.
        Thought: I need to find the watch first. I should go to shelf 1 to check and locate a watch.
        Action: go to shelf 1

        Observation: You arrive at shelf 1. On the shelf 1, you see nothing.
        Thought: There is no watch on shelf 1, I should check another locations.
        Action: go to tvstand 1
        
        Observation: You arrive at tvstand 1. On the tvstand 1, you see newspaper 1, a watch 1, and a remotecontrol 2.
        Thought: I have found a watch, I should pick it up now.
        Action: take watch 1 from tvstand 1

        Observation: You pick up the watch 2 from the tvstand 1.
        Thought: I now have watch 1. I need to go to a shelf and place watch on shelf.
        Action: go to shelf 2

        Observation: You arrive at shelf 2. On the shelf 2, you see nothing.
        Thought: I need to place watch 1 on shelf 2.
        Action: move watch 1 to shelf 2.

        goal completed!
        """