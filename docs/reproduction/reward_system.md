# Reward and episode termination

`SC2RawEnv.step` converts the PySC2 timestep reward to `float` and uses
`timestep.last()` for `done`. Normal game outcomes are `-1` loss, `0` draw, and
`+1` win. `tie_to_lose=True` converts terminal zero to `-1`; PPO training uses
that option, while evaluation leaves it off. The runtime log prints the actual
terminal difficulty and outcome.

Reward shaping is disabled by default. `train_ppo --use_reward_shaping` wraps
the raw environment in `KillingRewardWrapper`, which adds
`1e-5 * delta(killed_minerals + killed_vespene)` on non-terminal steps and
leaves the terminal outcome intact. The file also contains historical V1/V2
unit-count shaping wrappers, but the current PPO entry point does not select
them. Reward wrappers pass observations through unchanged.

Validated real-game results:

- random agent run 1: 577 decision steps, terminal `+1`, normal completion;
- random agent run 2 through the delivered script: 899 decision steps,
  terminal `0`, normal completion;
- one-update PPO checkpoint evaluation: 899 decision steps, terminal `0`,
  normal completion.

The displayed `(return + 1) / 2` evaluator value is meaningful as win rate only
for unshaped `-1/0/+1` outcomes. Shaped returns must not be reported as wins.
