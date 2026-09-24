# Objective

The agent's externally specified objective is:

```text
PLAY → LEARN → LIVE → GROW
```

We provide Minecraft mechanics and numerical observations, not a Minecraft strategy.

The environment may expose:

- all item/block/entity IDs
- recipes and transformations
- inventory/equipment structures
- entity state
- block state
- dimensions/locations
- available actions
- deterministic game mechanics

The agent learns:

- usefulness/value
- strategy
- resource priorities
- survival behavior
- exploration
- crafting behavior
- combat
- navigation
- reusable skills
- long-horizon planning
- adaptation

A Minecraft death is an environment termination event. Learned parameters, replay, world-model state outside the episode, and checkpoints persist.
