import re


def extract_picked_object(text):
    match = re.search(r"pick up the ([\w\s\d]+?) from", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_moved_object(text):
    match = re.search(r"you move (.+?) to", text, re.IGNORECASE)
    if match:
        obj = match.group(1).strip()
        return obj.replace("the ", "")  # removes "the" if present
    return None


def extract_locations(obs_text):
    lower_obs = obs_text.lower()
    pattern = r"\b(\w+\s\d+)\b"
    
    # Find all matches
    matches = re.findall(pattern, lower_obs)
    return matches


def parse_main_plan (Main_plan):
    pattern = r"\s*(\d+)\.\s+(.*?)\s*\[Thought\]:\s*(.*)"
    matches = re.findall(pattern, Main_plan)

    if not matches:
        return "No valid steps found."

    formatted_plan = ["Main Plan:"]
    for step_num, step_desc, thought in matches:
        formatted_plan.append(f"{step_num}. {step_desc.strip()}\n   Thought: {thought.strip()}\n")

    return "\n".join(formatted_plan)


def parse_executor(executor_text):
    pattern = r"(?i)thought:\s*(?P<thought>.+?)\n\s*action:\s*(?P<action>.+?)\n\s*summary:\s*(?P<summary>.+)"
    match = re.search(pattern, executor_text.strip(), re.DOTALL)

    if not match:
        return None, None, None

    thought = match.group("thought").strip()
    action = match.group("action").strip()
    summary = match.group("summary").strip()

    return thought, action, summary


def task_type_instuction(task_type):
    if task_type == "examine_in_light":
        return"""
        Examine in Light 
        - The agent must locate an object of the desired type first, pick it up, then locate and turn on a light source (most likely a lamp) with the desired object in-hand. 
        """.strip()
    
    elif task_type == "clean_&_place":
        return"""
        Clean & Place 
        - The agent must locate an object of the desired type, pick it up, head to a sink or basin, and while holding it, perform the action "clean (object) with (receptacle)", using the sink or basin as the receptacle to clean the item. Once cleaned, the agent should locate the correct placement destination and move the object there.
        """.strip()
    
    elif task_type == "cool_&_place":
        return"""
        Cool & Place 
        - The agent must locate an object of the desired type, pick it up, go to a fridge, open it, and while holding the object, perform the action "cool (object) with (receptacle)", using the fridge as the receptacle to turn the item into its cooled variant. Once cooled, the agent should locate the correct placement destination and move the object there.
        """.strip()

    elif task_type == "heat_&_place":
        return"""
        Heat & Place  
        - The agent must locate an object of the desired type, pick it up, go to a microwave (the heating source must be microwave), open it and while holding the object, perform the action "heat (object) with (receptacle)" using the microwave as the receptacle to obtain the heated version. Once heated, the agent should locate the correct placement destination and move the object there.
        """.strip()
    
    elif task_type == "pick_two_&_place":
        return"""
        Pick Two & Place  
        - The agent can only carry one object at a time. Thus, the agent must locate an object of the desired type, pick it up, find the correct location to place it and place down the object there, then look for another object of the same desired type, pick it up, return to previous location, and place down the object there beside the previous placed object. There should be two object of the same desired type at the same placement destination. 
        """.strip()
    
    elif task_type == "pick_&_place":
        return"""
        Pick & Place
        - The agent must locate an object of the desired type, pick it up, find any placement destination to place it, and move it there.
        """.strip()
    else:
        return"""
        None.
        """.strip()
    

def extract_task_type(goal_text):
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
    