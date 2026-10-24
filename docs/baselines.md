# Reproducible baselines and execution matrix

## Validated core baseline

```bash
python -m pip install -e ".[test]" -c requirements/constraints-py311.txt
python -m pytest -m "not sc2"
python -m sc2learner.bin.train_ppo_local \
  --updates 10 --rollout-steps 128 --horizon 8 --seed 0 --device cpu
```

This CPython 3.11/PyTorch 2.x CPU run validates data flow; it is not an SC2
score baseline.

## PPO against StarCraft II

Preserved defaults are `gamma=0.998`, `lambda=0.95`, clip range `0.1`, entropy
coefficient `0.01`, value coefficient `0.5`, Adam epsilon `1e-5`, rollout
length `128`, learner batch size `32`, and global gradient norm `0.5`.

```bash
python -m sc2learner.bin.train_ppo \
  --job_name=learner --policy=mlp --device=cuda \
  --port_A=5700 --port_B=5701 --save_dir=checkpoints/ppo
python -m sc2learner.bin.train_ppo \
  --job_name=actor --policy=mlp --device=cpu \
  --learner_ip=127.0.0.1 --port_A=5700 --port_B=5701
```

Use `--policy=lstm` on both sides for the recurrent policy. Workers must agree
on policy, observation, action and port configuration.

## Self-play

```bash
python -m sc2learner.bin.train_ppo_selfplay --job_name=learner --policy=mlp --device=cuda
python -m sc2learner.bin.train_ppo_selfplay \
  --job_name=actor --policy=mlp --device=cpu --learner_ip=127.0.0.1
```

The default pool holds 300 named parameter snapshots and caches a learner
snapshot with probability `0.05`. An initial pool file contains one native
PyTorch checkpoint path per line. TensorFlow/joblib lists are not accepted.

## DQN

The preserved defaults include discount `0.995`, MMC beta `0.9`, Adam epsilon
`1e-7`, per-parameter gradient clamp `10`, target interval `10000`, and the
original two-stage epsilon formula.

```bash
python -m sc2learner.bin.train_dqn --job_name=learner --device=cuda
python -m sc2learner.bin.train_dqn --job_name=actor --device=cpu --learner_ip=127.0.0.1
```

## Validation and security boundary

Core, recurrent, DQN numerical, checkpoint, opponent-pool and local ZeroMQ
loopback tests run without SC2. Real games require the Tencent extension, SC2
4.6 data and AbyssalReef. No game win rate is claimed here.

ZeroMQ messages use Python pickle and have no authentication. Bind learner
ports only on a trusted network or protect them with an external secure tunnel
or firewall.
