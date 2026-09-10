import json, pyrosetta
pyrosetta.init("-mute all")
p = pyrosetta.pose_from_pdb("/home/eduardo/Insulin_project/ensemble_sampling/inputs/f3_tm_constricted.pdb")
info = p.pdb_info()

# From the YAML: replace with your actual chain + residue numbers
F3_N, F3_C = info.pdb2pose("B", 91), info.pdb2pose("B", 183)
TM_N, TM_C = info.pdb2pose("B", 198),  info.pdb2pose("B", 218)

for label, i in [("F3 N-term CA", F3_N), ("F3 C-term CA (=anchor)", F3_C),
                 ("TM N-term CA (=placed)", TM_N), ("TM C-term CA", TM_C)]:
    ca = p.residue(i).xyz("CA")
    print(f"{label:24s} z={ca.z:+7.2f}  xy=({ca.x:+.2f},{ca.y:+.2f})")

m = json.load(open("/home/eduardo/Insulin_project/ensemble_sampling/trial_grid_z-20/metrics.json"))
print("stored anchor_position:", m["grid"][list(m["grid"])[0]]["anchor_position"])