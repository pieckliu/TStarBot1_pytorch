"""Run the migrated PPO path locally without StarCraft II."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sc2learner.agents.ppo_agent import Model
from sc2learner.agents.ppo_local import LocalPPOTrainer, evaluate_policy
from sc2learner.agents.ppo_policies import MlpPolicy
from sc2learner.envs.mock_env import SyntheticMacroEnv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args(argv)


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = SyntheticMacroEnv(masked=True, horizon=args.horizon)
    model = Model(
        policy=MlpPolicy,
        ob_space=env.observation_space,
        ac_space=env.action_space,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    trainer = LocalPPOTrainer(
        env,
        model,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
    )
    history = []
    for update in range(1, args.updates + 1):
        metrics, _ = trainer.update(seed=args.seed + update)
        metrics["update"] = update
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
    evaluation = evaluate_policy(env, model, episodes=4, seed=args.seed + 10_000)
    summary = {
        "updates": args.updates,
        "evaluation_mean_return": evaluation["mean_return"],
        "final_metrics": history[-1] if history else {},
    }
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.checkpoint)
        summary["checkpoint"] = str(args.checkpoint)
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.updates < 1:
        raise SystemExit("--updates must be positive")
    run(args)


if __name__ == "__main__":
    main()
