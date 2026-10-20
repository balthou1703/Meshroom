# mrDccPlugin

Meshroom plugin giving the **Houdini**, **Maya** and **Nuke** contexts of the Meshroom subgraphs, and nodes working
on their scenes. A `SubGraph` node of Meshroom whose nodes need one of these contexts runs as one task in one session
of the software, and the same subgraph runs from the Python shell of the software, in its open session
(`meshroom.subgraph.run`). See `docs/subgraph.md` in Meshroom.

Requires a Meshroom with the subgraphs (`meshroom.core.subgraph`).

## Install

```bash
export MESHROOM_PLUGINS_PATH=/path/to/mrDccPlugin:$MESHROOM_PLUGINS_PATH
```

## Contexts

| Context | Session | Interpreter (task, Run Live) | Override |
|---------|---------|------------------------------|----------|
| `houdini` | `hou` (`ctx.context.session`) | `hython` | `MESHROOM_HYTHON` |
| `maya` | `maya.cmds` | `mayapy` | `MESHROOM_MAYAPY` |
| `nuke` | `nuke` | `nuke -t` | `MESHROOM_NUKE` |

A task node of another plugin whose `pythonExecutable` is `hython`, `mayapy` or `nuke` needs that context: a
subgraph containing it starts like that node would (its interpreter, the process environment of its plugin), unless
the override variable is set. The interpreter must be able to import Meshroom and its dependencies (e.g. `psutil`).

Scenes: the nodes written to load and save the scene themselves keep working as they are (`hou.hipFile.load` /
`save`, `cmds.file(open=...)` / `cmds.file(save=...)`, `nuke.scriptOpen` / `scriptSave` / `scriptSaveAs`): during a
subgraph, loading the scene already open does nothing, and their saves are grouped into one save at the end of a
Meshroom task (or at a Save node). In the session of a software, a node never replaces the open scene and nothing is
saved unless a Save node asks for it. The Maya nodes written for mayapy can call `maya.standalone.initialize` /
`uninitialize`: they do nothing during a subgraph.

In a Houdini session, another plugin importing `pxr` may load another USD than the one of Houdini (two USD
libraries in one process make Houdini abort): `import pxr.Tf` in Houdini first (the USD of Houdini), before
`meshroom.subgraph.run` loads the plugins.

## Nodes

| Node | Role |
|------|------|
| `HoudiniScene`, `MayaScene`, `NukeScene` | The scene, the value that prevails: in a Meshroom task, opens `Scene File` (e.g. given by a pipeline node); in a session of the software, the open scene (`Scene File` is not read), unless `In ... Session` is `sceneFile`. |
| `HoudiniSave`, `MayaSave`, `NukeSave` | Saves the scene once. In a session of the software, only if `Save In ... Session` is on. |
| `HoudiniSetParms` | Sets parameters of the Houdini scene (one per line, or from a connected value). |
| `HoudiniRopRender` | Renders / exports with a ROP: the frame of the loop, else the frame range of the ROP. |

Example: [meshroom/houdiniRender.mgt](meshroom/houdiniRender.mgt) — the scene, parameters, render of the frame of the loop.

```python
# Houdini Python shell (Meshroom in PYTHONPATH, MESHROOM_PLUGINS_PATH with this plugin)
import meshroom.subgraph
outputs = meshroom.subgraph.run("/path/to/mrDccPlugin/meshroom/houdiniRender.mgt", frames=(1001, 1010))
```

## Tests

```bash
PYTHONPATH=/path/to/Meshroom pytest tests
```
