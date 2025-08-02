from openai import OpenAI


llm_url = "http://172.21.47.110" # Set this to your LLM URL, e.g., "http://localhost:8000"
llm_port = "50963" # Set this to your LLM port, e.g., "8000"
chat_endpoint = "/v1" # Adjust if necessary
chat_url = f"{llm_url}:{llm_port}{chat_endpoint}"
client = OpenAI(base_url=chat_url, api_key="-")
llm_model = "qwen2.5:32b-32k" # Set this to your LLM model, e.g., "gpt-3.5-turbo"


def chat_with_llm(prompt, model=None, temperature=0.2):
    # Use passed-in model or fall back to default
    chosen_model = model if model is not None else llm_model

    # Create message payload
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        response = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content, response.usage.dict()
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        # Return consistent structure for failure case
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def parse_response(response):
    if isinstance(response, tuple):
        response = response[0]
    lines = response.strip().split('\n')
    action = ""
    thought = ""

    for line in lines:
        if line.strip().lower().startswith("thought:") or line.strip().lower().startswith("Thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("action:") or line.strip().lower().startswith("Action:"):
            action = line.split(":", 1)[1].strip()

    # if not action:
    #     print("[WARNING] No action detected. Defaulting to 'look'.")
    #     action = "look"

    return action, thought


def extract_goal(obs_text):
    for line in obs_text.split('\n\n'):
        if "Your task is to:" in line:
            return line.split("Your task is to:")[-1].strip()
    return None


def process_step_inputs(obs, infos):
    """
    Extract observation text and admissible commands.
    """
    # Extract observation
    if isinstance(obs, (list, tuple)) and len(obs) > 0:
        current_obs_text = str(obs[0])
    else:
        current_obs_text = str(obs)

    # Extract admissible commands
    cmds = infos.get("admissible_commands", [])
    if isinstance(cmds, (list, tuple)) and len(cmds) > 0 and isinstance(cmds[0], (list, tuple)):
        cmds = list(cmds[0])
    return current_obs_text, cmds