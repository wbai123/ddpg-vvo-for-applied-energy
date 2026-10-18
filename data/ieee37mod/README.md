# Modified IEEE 37-bus case

`ieee37.dss` is the working case used by the simulator.
`ieee37.mod-src.dss` preserves the supervisor’s original file unchanged for reference.

## Changes from the supervisor source

The circuit topology, lines, both regulator banks and nominal load definitions are unchanged.

- Removed the duplicate x2 load definitions and the commented-out x0.5 block. Experiments apply dataset multipliers to the nominal loads.
- Removed the manual tap ratios of 1.3 and `Set Controlmode=OFF`. The simulator manages tap settings and control mode for each experiment.
