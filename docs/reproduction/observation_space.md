# Runtime observation space

With the validated MLP flags (`use_spatial_features=false`, game progress on,
8-action history, regions off, action mask on), reset and every step return:

```text
Tuple(Box(0, inf, (857,), float32), Box(0, 1, (62,), float32))
```

The mask is intentionally present twice: once appended to the 857-vector and
once as the second tuple element consumed by the masked policy.

| Feature | Dimensions | Source and normalization |
|---|---:|---|
| unit type counts | 76 | 19 unit types x self/enemy x scaled/log; count `/20` and `log10(count+1)` |
| building counts | 64 | 16 building types x self/enemy x scaled/log; same normalization |
| aggregate unit statistics | 16 | self/enemy total, combat, ground, air; `/20` and `log10(count+1)` |
| player | 28 | 9 PySC2 player values as scaled and log features plus 10-bin unused-supply one-hot |
| cumulative score | 20 | 10 score entries `/3000`, then `log10(scaled+1)` |
| workers | 3 | gas, mineral, other drone counts, each `/20` |
| action history | 496 | 8 one-hot action IDs x 62 actions |
| game progress | 92 | game loop one-hot at 60, 20, 8, and 4 resolutions over 24,000 loops |
| embedded action mask | 62 | current TechTree/state validity, float32 0/1 |
| **Total** | **857** | concatenated float32 vector |

Only raw units supplied by PySC2 are counted. The validated run keeps fog of
war enabled, so enemy features represent currently observable enemy units, not
hidden omniscient state. Spawn-side flipping canonicalizes fixed regions. Reset
and step shapes were exercised by both the random full game and PPO rollout.

Optional region features multiply unit/stat counts across the ten fixed map
regions; optional spatial mode adds unit/alliance minimap channels. Those modes
were inspected in source but were not part of this MLP smoke run.
