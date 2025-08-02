from alfworld.agents.environment import get_environment
from goalie import Goalie
from utils.agent_utils import extract_goal
from datetime import datetime
import yaml
import json
import os


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("goalie/goalie_log", exist_ok=True)
log_file_path = f"goalie/goalie_log/chat_log_{timestamp}.jsonl"


def log_episode_jsonl(episode, goal, success_rate, step_count, avg_step, completed=None, history=None, main_plan=None, usage_summary=None):
    entry = {
        "episode": episode,
        "goal": goal,
        "main_plan": main_plan,
        "success_rate": round(success_rate, 2),
        "steps": step_count,
        "avg_step": round(avg_step, 2),
        "completed": completed,
        "history": history
    }

    if usage_summary:
        entry["token_usage"] = usage_summary

    with open(log_file_path, "a") as log_file:
        log_file.write(json.dumps(entry, indent=2) + "\n")


def log_cumulative_jsonl(success_rate, avg_steps, usage_summary, goals):
    entry = {
        "Cumulative Summary": True,
        "success_rate": round(success_rate, 2),
        "avg_step": round(avg_steps, 2),
        "token_usage": usage_summary,
        "cumulative_goals": [
            {
                "goal": g["goal"],
                "status": "SUCCESS" if g["success"] else "FAILURE",
                "steps": g["steps"]
            } for g in goals
        ]
    }

    with open(log_file_path, "a") as log_file:
        log_file.write(json.dumps(entry, indent=2) + "\n")


def play():
    config_file = "configs/eval_config.yaml" # Adjust the path as needed
    max_steps_per_episode = 50
    skip_episode = 0

    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded config from {config_file}")
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return
    
    # Dynamically get the environment class based on config["env"]["type"]
    EnvClass = get_environment(config["env"]["type"])
    environment = EnvClass(config, train_eval="train").init_env(batch_size=1)

    # Create and reset agent
    agent = Goalie()
    agent.reset(environment) 

    num_episodes_to_run = 30
    successful_episodes = 0
    success_rate = 0.0
    total_step_count = 0
    avg_step_count = 0.0

    cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cumulative_goals = []
 
    for episode_num in range(num_episodes_to_run):
        episode_index = episode_num + 1

        if episode_index <= skip_episode:
            print(f"\n--- Skipping Episode {episode_index} ---")

            environment.reset()
            
            fake_goal = f"(Skipped Goal {episode_index})"
            cumulative_goals.append({
                "goal": fake_goal,
                "success": False,
                "steps": 0
            })
            log_episode_jsonl(
                episode=episode_index,
                goal=fake_goal,
                main_plan=[],
                success_rate=success_rate,
                step_count=0,
                avg_step=avg_step_count,
                history=[],
                completed=False,
                usage_summary={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            continue

        print(f"\n--- Starting Episode {episode_index} ---")
        episode_usage = {key: 0 for key in cumulative_usage}
        obs, info = environment.reset()
        agent.reset(environment)

        done = False
        reward = 0.0
        step_count = 1

        obs_text = obs[0]
        goal = extract_goal(obs_text) or "unknown"


        while True:
            action, usage, main_plan = agent.act(info, reward, done, obs) 
            print(f"\n[{step_count}] → {action}\n")

            for key in episode_usage:
                value = usage.get(key, 0)
                episode_usage[key] += value
                cumulative_usage[key] += value


            obs, reward, done, info = environment.step([action]) 
            current_obs_text = obs[0]
            current_reward = reward[0] 
            current_done = done[0]     

            if step_count >= max_steps_per_episode:
                current_done = True
                current_reward = 0.0

            if current_done:
                if current_done:
                    is_success = current_reward == 1.0
                if is_success:
                    successful_episodes += 1

                total_step_count += step_count
                success_rate = (successful_episodes / episode_index) * 100
                avg_step_count = total_step_count / episode_index
                
                outcome = "SUCCESS" if is_success else "FAILURE"
                print(f"{outcome} | Episode {episode_index} | Success rate: {success_rate:.2f}% | Steps: {step_count} | Total episode avg step: {avg_step_count:.2f}")

                if step_count >= max_steps_per_episode and not is_success:
                    failure_reason = f"Failure: exceeded {max_steps_per_episode} steps"
                else:
                    failure_reason = "Goal completed" if is_success else "Failure: task unsuccessful"

                agent.history[-1].update({
                    "goal": goal,
                    "success": is_success,
                    "note": failure_reason
                })

                cumulative_goals.append({
                    "goal": goal,
                    "success": is_success,
                    "steps": step_count
                })

                log_episode_jsonl(
                    episode=episode_index,
                    goal=goal,
                    main_plan=main_plan,
                    success_rate=success_rate,
                    step_count=step_count,
                    avg_step=avg_step_count,
                    history=agent.history,
                    completed=is_success,
                    usage_summary=episode_usage
                )
                break

            step_count += 1
    
    usage_averages = {
        "average_prompt_tokens": cumulative_usage["prompt_tokens"] // num_episodes_to_run,
        "average_completion_tokens": cumulative_usage["completion_tokens"] // num_episodes_to_run,
        "average_total_tokens": cumulative_usage["total_tokens"] // num_episodes_to_run
    }

    log_cumulative_jsonl(
        success_rate=success_rate,
        avg_steps=avg_step_count,
        usage_summary={**cumulative_usage, **usage_averages},
        goals=cumulative_goals
    )

    print("\n--- All episodes finished ---")


if __name__ == "__main__":
    play()