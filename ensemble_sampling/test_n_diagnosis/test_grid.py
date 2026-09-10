"""Smoke-test the grid-mode config parsing and placement enumeration
without loading PyRosetta.
"""
import sys, types

# Stub out PyRosetta imports so we can import the module without it installed.
for name in [
    "pyrosetta",
    "pyrosetta.rosetta",
    "pyrosetta.rosetta.core",
    "pyrosetta.rosetta.core.kinematics",
    "pyrosetta.rosetta.core.pack",
    "pyrosetta.rosetta.core.pack.task",
    "pyrosetta.rosetta.core.scoring",
    "pyrosetta.rosetta.core.select",
    "pyrosetta.rosetta.core.select.residue_selector",
    "pyrosetta.rosetta.numeric",
    "pyrosetta.rosetta.protocols",
    "pyrosetta.rosetta.protocols.loops",
    "pyrosetta.rosetta.protocols.minimization_packing",
    "pyrosetta.rosetta.protocols.moves",
]:
    sys.modules[name] = types.ModuleType(name)

# Now fake the specific class/attr imports the module needs at import time.
pk = sys.modules["pyrosetta.rosetta.core.kinematics"]
pk.FoldTree = pk.MoveMap = object
tf = sys.modules["pyrosetta.rosetta.core.pack.task"]
class _Op: pass
tf.TaskFactory = object
tf.operation = _Op
sfs = sys.modules["pyrosetta.rosetta.core.scoring"]
sfs.ScoreType = object
rs = sys.modules["pyrosetta.rosetta.core.select.residue_selector"]
rs.NeighborhoodResidueSelector = rs.ResidueIndexSelector = object
sys.modules["pyrosetta.rosetta.numeric"].xyzVector_double_t = object
rig = sys.modules["pyrosetta.rosetta.protocols.rigid"] = types.ModuleType(
    "pyrosetta.rosetta.protocols.rigid")
rig.RigidBodyTransMover = rig.RigidBodyDeterministicSpinMover = object
lp = sys.modules["pyrosetta.rosetta.protocols.loops"]
lp.Loop = lp.Loops = object
sys.modules["pyrosetta.rosetta.protocols.minimization_packing"].PackRotamersMover = object
sys.modules["pyrosetta.rosetta.protocols.moves"].MonteCarlo = object

sys.path.insert(0, "/home/claude")
import ensemble_sampling as es

# --- Test 1: back-compat -- random-mode YAML parses as before ---
random_cfg = {
    "segments": [
        {"name": "fn3",    "kind": "rigid",    "start": 1,  "stop": 95},
        {"name": "linker", "kind": "flexible", "start": 96, "stop": 113},
        {"name": "tm",     "kind": "rigid",    "start": 114,"stop": 136,
         "dof": {"translate": {"x": 15.0, "y": 15.0, "z": 4.0},
                 "tilt": 20.0, "spin": 180.0}},
    ]
}
c = es.Construct([es.Segment(name=e["name"], kind=e["kind"],
                             start=e["start"], stop=e["stop"],
                             dof=es.DOF.from_dict(e.get("dof")))
                  for e in random_cfg["segments"]])
c.validate()
assert c.rigid[1].dof.mode == "random"
assert c.rigid[1].dof.tx == 15.0
assert c.rigid[1].dof.is_mobile
print("Test 1 (random back-compat)                 : OK")

# --- Test 2: grid YAML parses and enumerates the expected grid ---
grid_cfg = {
    "segments": [
        {"name": "fn3",    "kind": "rigid",    "start": 1,  "stop": 95},
        {"name": "linker", "kind": "flexible", "start": 96, "stop": 113},
        {"name": "tm",     "kind": "rigid",    "start": 114,"stop": 136,
         "dof": {"mode": "grid", "anchor": "fn3",
                 "translate": {
                     "x": {"min": -4, "max": 4, "step": 2},
                     "y": {"min": -4, "max": 4, "step": 2},
                     "z": -15.0,
                 },
                 "tilt": 0.0, "spin": 0.0}},
    ]
}
c = es.Construct([es.Segment(name=e["name"], kind=e["kind"],
                             start=e["start"], stop=e["stop"],
                             dof=es.DOF.from_dict(e.get("dof")))
                  for e in grid_cfg["segments"]])
c.validate()
tm = c.rigid[1]
assert tm.dof.mode == "grid"
assert tm.dof.anchor == "fn3"
assert tm.dof.tz == -15.0                   # signed scalar
assert tm.dof.grid_axes == {"x": (-4.0, 4.0, 2.0),
                            "y": (-4.0, 4.0, 2.0)}
assert es._axis_values(-4, 4, 2) == [-4.0, -2.0, 0.0, 2.0, 4.0]
mobiles = [s for s in c.rigid if s.dof.is_mobile]
placements = list(es.enumerate_grid_placements(mobiles))
assert len(placements) == 5 * 5, f"expected 25 cells, got {len(placements)}"
# First cell: (x_min, y_min); last: (x_max, y_max); y varies fastest.
first, last = placements[0]["tm"], placements[-1]["tm"]
assert first["dx"] == -4.0 and first["dy"] == -4.0 and first["dz"] == -15.0
assert last["dx"] == 4.0 and last["dy"] == 4.0 and last["dz"] == -15.0
# y varies fastest -> placements[1] has dx=-4, dy=-2
assert placements[1]["tm"]["dy"] == -2.0
# grid indices ride along
assert placements[0]["tm"]["grid_x_i"] == 0.0
assert placements[0]["tm"]["grid_y_i"] == 0.0
assert placements[-1]["tm"]["grid_x_i"] == 4.0
assert placements[-1]["tm"]["grid_y_i"] == 4.0
# Deterministic tilt_azimuth in grid mode
assert placements[0]["tm"]["tilt_azimuth"] == 0.0
print("Test 2 (grid enumeration)                   : OK")

# --- Test 3: validation errors ---
def expect_error(dof_dict, msg_fragment, *, segments=None):
    segs = segments or [
        {"name": "fn3", "kind": "rigid", "start": 1, "stop": 95},
        {"name": "linker", "kind": "flexible", "start": 96, "stop": 113},
        {"name": "tm", "kind": "rigid", "start": 114, "stop": 136,
         "dof": dof_dict},
    ]
    try:
        c = es.Construct([es.Segment(name=e["name"], kind=e["kind"],
                                     start=e["start"], stop=e["stop"],
                                     dof=es.DOF.from_dict(e.get("dof")))
                          for e in segs])
        c.validate()
    except ValueError as e:
        assert msg_fragment in str(e), (
            f"expected {msg_fragment!r} in {e!r}")
    else:
        raise AssertionError(f"expected ValueError containing {msg_fragment!r}")

# Grid axis without mode: grid
expect_error({"translate": {"x": {"min": -4, "max": 4, "step": 2}}},
             "grid spec")
# Grid mode without anchor
expect_error({"mode": "grid",
              "translate": {"x": {"min": -4, "max": 4, "step": 2}}},
             "anchor")
# Grid mode with anchor pointing to a non-existent segment
expect_error({"mode": "grid", "anchor": "does_not_exist",
              "translate": {"x": {"min": -4, "max": 4, "step": 2}}},
             "not a declared rigid segment")
# Grid mode with self as anchor
expect_error({"mode": "grid", "anchor": "tm",
              "translate": {"x": {"min": -4, "max": 4, "step": 2}}},
             "cannot be self")
# Grid mode with anchor that is itself mobile: needs three rigids so we can
# make fn3 mobile via a grid spec (matching the outer mode) while tm anchors
# to it. A random-mode fn3 would trip the mixed-modes check first.
expect_error(
    {"mode": "grid", "anchor": "fn3",
     "translate": {"x": {"min": -4, "max": 4, "step": 2}}},
    "is mobile",
    segments=[
        {"name": "root",  "kind": "rigid", "start": 1, "stop": 50},
        {"name": "l0",    "kind": "flexible", "start": 51, "stop": 60},
        {"name": "fn3",   "kind": "rigid", "start": 61, "stop": 95,
         "dof": {"mode": "grid", "anchor": "root",
                 "translate": {"x": {"min": -2, "max": 2, "step": 2}}}},
        {"name": "linker", "kind": "flexible", "start": 96, "stop": 113},
        {"name": "tm",    "kind": "rigid", "start": 114, "stop": 136,
         "dof": {"mode": "grid", "anchor": "fn3",
                 "translate": {"x": {"min": -4, "max": 4, "step": 2}}}},
    ])
# Grid mode with only scalars (no grid axes)
expect_error({"mode": "grid", "anchor": "fn3",
              "translate": {"x": 5.0, "y": 5.0}},
             "at least one axis")
# Bad step (0)
expect_error({"mode": "grid", "anchor": "fn3",
              "translate": {"x": {"min": -4, "max": 4, "step": 0}}},
             "step must be > 0")
# max < min
expect_error({"mode": "grid", "anchor": "fn3",
              "translate": {"x": {"min": 4, "max": -4, "step": 2}}},
             "max < min")
# anchor in random mode
expect_error({"anchor": "fn3", "translate": {"x": 5.0}},
             "only meaningful in grid mode")
print("Test 3 (validation errors)                  : OK")

# --- Test 4: mixed modes rejected ---
try:
    segs = [
        {"name": "fn3", "kind": "rigid", "start": 1, "stop": 95},
        {"name": "linker1", "kind": "flexible", "start": 96, "stop": 105},
        {"name": "mid", "kind": "rigid", "start": 106, "stop": 130,
         "dof": {"translate": {"x": 5.0}}},   # random
        {"name": "linker2", "kind": "flexible", "start": 131, "stop": 140},
        {"name": "tm", "kind": "rigid", "start": 141, "stop": 165,
         "dof": {"mode": "grid", "anchor": "fn3",
                 "translate": {"x": {"min": -4, "max": 4, "step": 2}}}},
    ]
    c = es.Construct([es.Segment(name=e["name"], kind=e["kind"],
                                 start=e["start"], stop=e["stop"],
                                 dof=es.DOF.from_dict(e.get("dof")))
                      for e in segs])
    c.validate()
except ValueError as e:
    assert "same dof.mode" in str(e), str(e)
    print("Test 4 (mixed modes rejected)               : OK")
else:
    raise AssertionError("mixed modes should be rejected")

# --- Test 5: edge case -- single-cell grid ---
c = es.Construct([es.Segment(name="fn3", kind="rigid", start=1, stop=95),
                  es.Segment(name="linker", kind="flexible", start=96, stop=113),
                  es.Segment(name="tm", kind="rigid", start=114, stop=136,
                             dof=es.DOF.from_dict(
                                 {"mode": "grid", "anchor": "fn3",
                                  "translate": {"x": {"min": 0, "max": 0, "step": 2}},
                                  "tilt": 0.0, "spin": 0.0}))])
c.validate()
placements = list(es.enumerate_grid_placements([s for s in c.rigid if s.dof.is_mobile]))
assert len(placements) == 1, len(placements)
assert placements[0]["tm"]["dx"] == 0.0
print("Test 5 (single-cell grid)                   : OK")

print("\nAll tests passed.")