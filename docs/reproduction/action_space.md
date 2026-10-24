# Runtime action space

The validated default configuration uses `MaskDiscrete(62)`. This is the
runtime count, not the paper's historical 165-action figure. Enabling
`use_all_combat_actions` replaces the three default combat choices with ten
source-region-0 choices, producing 69 actions; it still does not produce 165.

## Registry

| IDs | Category | Names |
|---|---|---|
| 0 | no-op | `do_nothing` |
| 1-12 | building | `build_extractor`, `build_spawning_pool`, `build_roach_warren`, `build_hydraliskden`, `build_hatchery`, `build_evolution_chamber`, `build_baneling_nest`, `build_infestation_pit`, `build_spire`, `build_ultralisk_cavern`, `build_spine_crawler`, `build_spore_crawler` |
| 13-30 | production/morph | `morph_lurker_den`, `morph_lair`, `morph_hive`, `morph_greater_spire`, `produce_drone`, `produce_zergling`, `morph_baneling`, `produce_roach`, `morph_ravager`, `produce_hydralisk`, `morph_lurker`, `produce_mutalisk`, `produce_corruptor`, `morph_broodlord`, `produce_ultralisk`, `produce_overlord`, `morph_overseer`, `produce_queen` |
| 31-56 | technology | `upgrade_burrow`, `upgrade_centrifical_hooks`, `upgrade_chitions_plating`, `upgrade_evolve_grooved_spines`, `upgrade_evolve_muscular_augments`, `upgrade_gliare_constitution`, `upgrade_neural_parasite`, `upgrade_overlord_speed`, `upgrade_tunneling_claws`, flyer armor 1-3, flyer weapons 1-3, ground armor 1-3, zergling attack/move speed, melee weapons 1-3, missile weapons 1-3 |
| 57-58 | resources | `assign_workers_gather_gas`, `assign_workers_gather_minerals` |
| 59-61 | combat | `combats_in_region_0_attack_region_0`, `combats_in_region_9_attack_region_4`, `combats_in_region_4_attack_region_1` |

The real random-agent log prints the exact ID/name table and action counts for
all 62 entries.

## Selection and execution

The flow is `action_id -> ZergActionWrapper.step -> Function.function -> raw
SC2 Action list -> SC2RawEnv.step`. Build, produce, and upgrade validity checks
come from the 4.6 TechTree plus current resources, supply, prerequisites,
producer availability, maximum counts, and already-pending tasks. The
`float32[62]` mask has 1 for valid actions and 0 for invalid actions. The PPO
distribution assigns invalid logits negative infinity. Direct callers should
respect the mask; the wrapper does not reject an intentionally forced invalid
ID before calling its implementation. Most implementation functions safely
return an empty raw-action list when no producer or placement is available.

`do_nothing` itself returns an empty list. It does not disable the four
hand-coded persistent behaviors that run around every learned action:

- pre-actions: idle drones gather minerals; queens with at least 25 energy
  inject the nearest mature Hatchery/Lair/Hive;
- post-actions: newly seen combat units receive a rally task; saved attack
  tasks are executed every frame.

These are execution rules, not learned action choices.

## Combat behavior and map assumptions

`combat.py` defines one whole-map region and nine fixed Abyssal Reef rectangles
within coordinates `(0..200, 0..176)`. Spawn orientation flips region IDs.
An attack action records a target region in `_attack_tasks` for each selected
unit and emits no raw command immediately. The post-action then persists that
task: units attack prioritized visible targets in the region or attack-move to
fixed rally coordinates. Ravagers may cast Corrosive Bile; Lurkers burrow near
ground enemies and unburrow to move. Target selection and these micro rules are
hand-coded. No coordinates were changed during this work.

The real full-game random run executed building, production, technology,
resource-management, combat, and no-op categories and finished normally.
