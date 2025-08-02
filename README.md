# Goalie: Guiding Embodied Agents with Subgoal-Driven Prompts
Goalie is a lightweight, prompt-based framework that improves goal-directed behaviour in Large Language Model (LLM) agents within embodied environments. Designed for complex and long-horizon tasks, Goalie decomposes high-level goals into more manageable subgoals, maintains coherence with summarised histories, and detects derailments to enable adaptive recovery.

## Set-up
1. Clone this repository:
```
git clone https://github.com/vaisiyabalakrishnan/GOALIE
```
2. Install the module dependencies into your environment:
```
pip install -r requirements.txt
```

## Using Goalie
1. **Configure LLM:** Modify the `chat_with_llm()` function in the `llm_utils.py` file under the utils directory to use the LLM of your choice.
    * Set `llm_url`, `llm_port`, and `llm_model`.
2. **Set Parameters:** Modify the `play()` function in the `play_goalie.py` file under the goalie directory for your desired trials.
    * Edit `config_file`, `max_steps_per_episode`, and `num_episodes_to_run`if needed.
3. **Run Goalie:**
```
PYTHONPATH="${PYTHONPATH}:$(pwd)" python goalie/play_goalie.py
```
