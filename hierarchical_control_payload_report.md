# Hierarchical Control Payload Report

## Scope

This report explains why a prebuilt file such as `1_offline_train/1s_hierarchical_actions.csv`
is structurally valid but does not behave as a true time-replayed control payload in the current
`scenario-hierarchical-xangai-UAV` execution path.

It also describes what would need to change for this style of offline payload to work as expected.

No code changes were applied as part of this report.

## Short Answer

`hierarchical_actions.csv` is the correct control payload format, but in the current project the
rows are normally applied when the file is parsed, not when the per-row timestamp says they should
be applied.

As a result, a file containing many future actions is consumed all at once on the first control
read, and the later rows overwrite the earlier rows immediately.

## Why The Trap Happens

### 1. The file format is valid

The parser in `LteEnbNetDevice::ReadControlFile()` accepts the hierarchical format:

`timestamp,action_type,param1,param2`

For Energy Saving:

`timestamp,0,cellId,hoAllowed`

For Traffic Steering:

`timestamp,1,imsi,targetCellId`

That means a file like:

```csv
100,0,2,1
100,0,3,1
200,0,2,0
200,0,3,0
```

is syntactically correct.

### 2. The parser reads the whole file in one pass

In `src/lte/model/lte-enb-net-device.cc`, `ReadControlFile()` opens the control file and iterates
through every line with `while (std::getline(csv, line))`.

So if the file already contains actions for `100, 200, 300, ... 1000 ms`, all of them are read
in the same invocation.

### 3. The default mode does not schedule actions by row timestamp

The key behavioral switch is `LteEnbNetDevice::ScheduleControlMessages`.

Its default is:

`BooleanValue(false)`

When `m_scheduleControlMessages == false`, the hierarchical ES branch executes:

```cpp
m_rrc->SetSecondaryCellHandoverAllowedStatus(cellId, hoAllowed);
```

immediately during parsing.

So the timestamp column is parsed, but not used to delay execution.

### 4. Later rows overwrite earlier rows in the same read

Because all rows are applied immediately:

- the `100 ms` state is set
- then the `200 ms` state is set
- then the `300 ms` state is set
- and so on

By the time `ReadControlFile()` finishes, only the last row for each cell matters.

For `1_offline_train/1s_hierarchical_actions.csv`, that means the file behaves like:

"set cell 2 and cell 3 to their final listed state now"

not:

"replay this state sequence over ten control intervals"

### 5. The file is then flushed

At the end of `ReadControlFile()`, when `m_scheduleControlMessages == false`, the file is cleared:

```cpp
std::ofstream csvDelete{};
csvDelete.open(m_controlFilename.c_str());
```

So the queued actions are not preserved for later control periods.

This is why a prebuilt multi-timestamp file is a trap in the current default mode:

- all actions are consumed on the first read
- later control periods see an empty file

## Why This Did Not Break The RL / Semaphore Flow

The online RL flow does not rely on the file as a long timeline.

Instead, it works as a step-by-step handshake:

1. ns-3 emits metrics and blocks on the control semaphore
2. Python writes only the current step's actions to `hierarchical_actions.csv`
3. Python releases the control semaphore
4. ns-3 reads and applies the file immediately
5. ns-3 clears the file

This design is compatible with `ScheduleControlMessages=false` because each file write contains
only the action for the current step.

The trap appears when a user interprets the same file as an offline action timeline.

## How To Make Offline Timestamped Payloads Work

There are two clean ways to make prebuilt timestamped control payloads behave correctly.

### Option 1: Enable scheduled execution

Set:

`ns3::LteEnbNetDevice::ScheduleControlMessages = true`

Then `ReadControlFile()` will schedule each action at the row timestamp instead of applying it
immediately.

For the hierarchical ES branch, it already contains the scheduled path:

```cpp
Simulator::Schedule(MilliSeconds(timestamp),
                    &LteEnbRrc::SetSecondaryCellHandoverAllowedStatus,
                    m_rrc,
                    cellId,
                    hoAllowed);
```

This is the smallest semantic change because the parser already knows how to do it.

However, this mode changes the control model:

- the file becomes a preplanned timeline
- not an interactive step-by-step controller

That means it is well suited for offline replay, but not ideal for the current semaphore-driven RL
loop.

### Option 2: Feed only one time slice per control interval

Keep `ScheduleControlMessages=false`, but change the external writer so that each control period
only writes the actions for that specific timestamp.

Example:

- at `100 ms`, write only the `100 ms` rows
- at `200 ms`, replace the file with only the `200 ms` rows
- and so on

This preserves the current control semantics and fits the existing semaphore handshake.

It is the better choice if the goal is to mimic what the RL environment already does.

## Practical Evaluation Of `1_offline_train/1s_hierarchical_actions.csv`

### What is good about it

- It has the right number of columns.
- It uses valid hierarchical ES rows.
- It uses binary `hoAllowed` values.
- It is suitable as a source dataset for offline replay logic.

### What is not sufficient about it

- It contains multiple timestamps in one file.
- The current default control path consumes all rows at once.
- It only addresses cells `2` and `3`, leaving cells `4..8` untouched.

### What it can be used for right now

Without changing simulator behavior, it can be used as:

- a valid one-shot ES control file
- a source file to be sliced into per-step payloads
- a test artifact for parser validation

It cannot be used as a true timed replay file in the current default mode.

## Recommended Change Directions

If the objective is offline reproduction of a timeline:

- enable `ScheduleControlMessages=true`
- run with `useSemaphores=0`
- pass a prebuilt `hierarchical_actions.csv`

If the objective is compatibility with the current interactive RL design:

- keep `ScheduleControlMessages=false`
- keep the semaphore flow
- use a wrapper that writes one timestamp batch at a time

## Changes I Would Do

No changes were applied, but these are the changes I would make depending on the goal.

### Minimal simulator-side change for offline replay

1. Expose `ScheduleControlMessages` in the UAV scenario as a command-line/global parameter.
2. Set `Config::SetDefault("ns3::LteEnbNetDevice::ScheduleControlMessages", BooleanValue(...))`.
3. Use `useSemaphores=0` when replaying a prebuilt offline timeline.
4. Pass `--controlFileName=hierarchical_actions.csv`.

This would make `1_offline_train/1s_hierarchical_actions.csv` replay over time as intended.

### Minimal wrapper-side change without touching simulator semantics

1. Keep `ScheduleControlMessages=false`.
2. Build a small shell or Python wrapper that:
   - reads the full offline file
   - groups rows by timestamp
   - writes only the current timestamp batch into `hierarchical_actions.csv`
   - releases the control semaphore
3. Repeat until the file is exhausted.

This would preserve the current online control design while allowing reuse of offline training
payloads.

### Longer-term clean design

1. Separate the two modes explicitly:
   - `interactive control mode`
   - `offline replay mode`
2. Document the semantics of `hierarchical_actions.csv` for each mode.
3. Add a dedicated runner for offline replay so users do not accidentally feed a full timeline into
   the interactive parser.

