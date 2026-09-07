"""
Script for running AF3 with a given template.

INPUTS:
- Fasta containing the subunits and proper headers
- Template CIF files

FASTA HEADER CONVENTION (adjust to taste, just keep parse_fasta_to_df in sync):
    >file_ID|chain_id|template_cif_path|template_chain_id|template_original_cif_path
    SEQUENCE...

    - file_ID:            groups records into one AF3 job / one output json
    - chain_id:            the AF3 "id" for this entity, e.g. A, B, C, D
    - template_cif_path:   path to the template mmCIF for this entity
                            (leave empty to skip templating this entity)
    - template_chain_id:   chain ID *inside that cif* to use as the template
    - template_original_cif_path: OPTIONAL. Path to the original,
                            unextracted PDB cif template_cif_path was pulled
                            from. If given, release-date header metadata gets
                            grafted from this file onto template_cif_path
                            before use -- needed because single-chain
                            extraction commonly drops it, and AF3 refuses
                            templates without one ("The structure must have
                            a release date."). Leave blank if
                            template_cif_path already has it.

Records sharing a file_ID are combined into one job:
    1 record  -> generate_templated_AF_json_monomer
    2 records -> generate_templated_AF_json_protomer
    4 records -> generate_templated_AF_json_dimer

All three delegate to the same underlying builder (_generate_templated_AF_json)
so the template/alignment logic only lives in one place.
"""

### IMPORTS ################################################
import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path

import gemmi
import pandas as pd

### FUNCTIONS ##############################################


def sequence_extractor(cif_path, chain_id: str) -> tuple[str, list[int]]:
    """
    Extract the RESOLVED-only sequence of one chain from a template mmCIF,
    i.e. only residues that actually have coordinates.

    Returns:
        resolved_sequence: one-letter code string, resolved residues only,
            in chain order.
        full_index_lookup: 0-based positions within resolved_sequence itself
            (0, 1, 2, ...). Confirmed empirically against a real failure:
            AF3's own template coordinate array (`positions` in
            get_polymer_features) is built by walking only resolved
            residues in order -- its length exactly matched this script's
            resolved-residue count, and a label_seq-based index (which
            counts unresolved residues too, per AF3's *docs*) overshot it
            with an IndexError as soon as the template had any unresolved
            gap. Since resolved_sequence is built the same way (resolved
            residues, in order, nothing else), plain 0-based indices are
            actually the *exact* correspondence needed here, not just a
            fallback -- position i always means "the i-th resolved residue"
            on both sides, gaps notwithstanding.
    """
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()

    chain_id = chain_id.strip()
    try:
        chain = st[0][chain_id]
    except (IndexError, KeyError, ValueError):
        available = [c.name for c in st[0]]
        raise ValueError(
            f"Chain '{chain_id}' not found in {cif_path}. "
            f"Chains present: {available}"
        )

    polymer = chain.get_polymer()
    resolved_sequence = gemmi.one_letter_code(polymer.extract_sequence())

    if not resolved_sequence:
        raise ValueError(
            f"No resolved polymer residues found for chain '{chain_id}' in "
            f"{cif_path} -- wrong chain ID, or a non-polymer/hetero chain?"
        )

    full_index_lookup = list(range(len(resolved_sequence)))

    return resolved_sequence, full_index_lookup


def aligner(
    query_sequence: str,
    resolved_template_sequence: str,
    full_index_lookup: list[int],
    scoring: "gemmi.AlignmentScoring | None" = None,
) -> tuple[list[int], list[int]]:
    """
    Global-align query_sequence against a template's resolved-only sequence,
    and return AF3-ready (queryIndices, templateIndices) -- 0-based,
    parallel, gap-free lists over the *aligned* columns only.

    This is a genuine sequence alignment (gemmi's ksw2-based global aligner,
    BLOSUM62 by default), so query and template don't need to be identical
    length or identity -- mismatches are kept (AF3 allows templating from
    homologs/mutants). If you want strict-identity-only positions, inspect
    `gemmi.align_string_sequences(...).match_string` yourself and filter.

    Passing the FULL resolved_template_sequence (rather than some sub-window
    of it) is exactly how you "use the whole cif" as a template -- no manual
    windowing needed, the aligner finds the right region itself.
    """
    if scoring is None:
        scoring = gemmi.AlignmentScoring("b")  # BLOSUM62, standard for protein

    q_list = gemmi.expand_one_letter_sequence(query_sequence, gemmi.ResidueKind.AA)
    t_list = gemmi.expand_one_letter_sequence(
        resolved_template_sequence, gemmi.ResidueKind.AA
    )
    result = gemmi.align_string_sequences(q_list, t_list, [], scoring)

    query_indices: list[int] = []
    template_indices: list[int] = []
    q_pos, t_pos = 0, 0

    for length_str, op in re.findall(r"(\d+)([MID])", result.cigar_str()):
        length = int(length_str)
        if op == "M":  # aligned column (match or mismatch)
            for k in range(length):
                query_indices.append(q_pos + k)
                template_indices.append(full_index_lookup[t_pos + k])
            q_pos += length
            t_pos += length
        elif op == "I":  # residue in query, none in template -> skip
            q_pos += length
        elif op == "D":  # residue in template, none in query -> skip
            t_pos += length

    if not query_indices:
        raise ValueError(
            "Alignment produced zero matched positions -- check that the "
            "right chain/template was supplied."
        )

    coverage = len(query_indices) / len(query_sequence)
    print(
        f"  [aligner] {len(query_indices)} aligned residues, "
        f"{coverage:.0%} of query covered, "
        f"identity={result.calculate_identity():.1f}%"
    )

    return query_indices, template_indices


def _read_cif_text(cif_path) -> str:
    """Read a cif file as plain text, transparently handling gzip (detected
    by magic bytes, not filename, since your local cifs are plain .cif)."""
    cif_path = Path(cif_path)
    with open(cif_path, "rb") as f:
        is_gzip = f.read(2) == b"\x1f\x8b"
    if is_gzip:
        with gzip.open(cif_path, "rt") as f:
            return f.read()
    return cif_path.read_text()


# Header categories that carry a structure's release/deposition date. AF3
# raises "The structure must have a release date." if these are missing --
# common after single-chain extraction, since many extraction tools keep
# _atom_site but drop surrounding header categories. Grafted wholesale from
# a real file rather than hand-typed, since a hand-typed attempt (wrong loop
# vs non-loop formatting is a likely culprit) is a known way this can still
# fail silently -- see github.com/google-deepmind/alphafold3/issues/416.
_RELEASE_DATE_CATEGORIES = [
    "_pdbx_database_status.",
    "_pdbx_audit_revision_history.",
]


def graft_release_date_metadata(extracted_cif_path, original_cif_path, output_path) -> list[str]:
    """
    Copy release-date-bearing header categories from an original, unmodified
    PDB mmCIF onto a single-chain-extracted cif that's missing them. Returns
    the list of categories actually found and copied (empty list means the
    original didn't have them either -- worth knowing before running AF3).
    """
    orig_block = gemmi.cif.read(str(original_cif_path)).sole_block()
    doc = gemmi.cif.read(str(extracted_cif_path))
    block = doc.sole_block()

    grafted = []
    for prefix in _RELEASE_DATE_CATEGORIES:
        table = orig_block.find_mmcif_category(prefix)
        if not table or table.width() == 0:
            continue
        short_tags = [t[len(prefix):] for t in table.tags]
        columns = {tag: [] for tag in short_tags}
        for row in table:
            for tag, val in zip(short_tags, row):
                columns[tag].append(val)
        block.set_mmcif_category(prefix, columns)
        grafted.append(prefix)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.write_file(str(output_path))
    print(f"  [graft_release_date_metadata] {Path(extracted_cif_path).name}: "
          f"copied {grafted or 'NOTHING -- original lacks these too'} "
          f"from {original_cif_path}")
    return grafted


def _build_template_entry(query_sequence: str, template_cif_path, template_chain_id: str, schema_version: int) -> dict:
    """Combine sequence_extractor + aligner into one AF3 'templates' entry."""
    resolved_seq, full_index_lookup = sequence_extractor(
        template_cif_path, template_chain_id
    )
    query_indices, template_indices = aligner(
        query_sequence, resolved_seq, full_index_lookup
    )
    entry = {"queryIndices": query_indices, "templateIndices": template_indices}
    if schema_version >= 2:
        # mmcifPath only exists from schema version 2 onwards.
        entry["mmcifPath"] = str(Path(template_cif_path).resolve())
    else:
        entry["mmcif"] = _read_cif_text(template_cif_path)
    return entry


def _entity_block(row, grafted_cif_dir, schema_version: int) -> dict:
    """Build one 'sequences' list entry (one protein entity) from a row."""
    chain_id = row["chain_id"]
    seq = row["sequence"]

    templates = []
    template_cif_path = row.get("template_cif_path") if hasattr(row, "get") else row["template_cif_path"]
    if template_cif_path and str(template_cif_path).strip() and pd.notna(template_cif_path):
        original_cif_path = row.get("template_original_cif_path") if hasattr(row, "get") else row.get("template_original_cif_path", "")
        if original_cif_path and str(original_cif_path).strip() and pd.notna(original_cif_path):
            grafted_path = Path(grafted_cif_dir) / f"{chain_id}_{Path(template_cif_path).stem}.cif"
            graft_release_date_metadata(template_cif_path, original_cif_path, grafted_path)
            template_cif_path = grafted_path
        print(f"Aligning template for chain {chain_id}...")
        templates.append(
            _build_template_entry(seq, template_cif_path, row["template_chain_id"], schema_version)
        )

    return {
        "protein": {
            "id": chain_id,
            "sequence": seq,
            "unpairedMsa": f">dummy\n{seq}\n",
            "pairedMsa": "",
            "templates": templates,
        }
    }


def _generate_templated_AF_json(rows, json_path, file_name: str, schema_version: int = 1):
    """Shared builder behind the monomer/protomer/dimer wrappers below.

    schema_version defaults to 1 -- confirmed against an actual traceback
    ("AlphaFold 3 input JSON has unsupported version: 2, expected 1") that
    this workstation's AF3 build predates the versioned-schema feature
    entirely. Pass schema_version=2+ if/when that install is updated.
    """
    json_data = {
        "name": file_name,
        "sequences": [
            _entity_block(row, Path(json_path) / "grafted_templates", schema_version)
            for row in rows
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": schema_version,
    }

    out_path = Path(json_path) / f"{file_name}.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Wrote {out_path}")
    return out_path


def generate_templated_AF_json_monomer(row, json_path, schema_version: int = 1):
    file_name = row["file_ID"]
    return _generate_templated_AF_json([row], json_path, file_name, schema_version)


def generate_templated_AF_json_protomer(rows, json_path, schema_version: int = 1):
    # There will be 2 protein entities, so 2 template sections, 2 cif input
    # files and 2 aligner runs (one per protomer) -- handled by _entity_block
    # being called once per row in _generate_templated_AF_json.
    rows = list(rows)
    assert len(rows) == 2, f"protomer expects 2 rows, got {len(rows)}"
    file_name = rows[0]["file_ID"]
    return _generate_templated_AF_json(rows, json_path, file_name, schema_version)


def generate_templated_AF_json_dimer(rows, json_path, schema_version: int = 1):
    # 4 protein entities -> 4 template sections, 4 cif files, 4 aligner runs.
    rows = list(rows)
    assert len(rows) == 4, f"dimer expects 4 rows, got {len(rows)}"
    file_name = rows[0]["file_ID"]
    return _generate_templated_AF_json(rows, json_path, file_name, schema_version)

def run_af(
    json_dir,
    output_dir,
    model_dir,
    db_dir,
    conda_env_path: str = "/home/ingemar/anaconda3/envs/alphafold3",
    af3_script_path: str = "/mnt/data/alphafold3/run_alphafold.py",
    num_diffusion_samples: int = 1,
    execute: bool = False,
):
    """
    Build (and optionally run) AF3 via the native/conda invocation that
    works on this workstation -- one call per generated json file, since
    --json_path takes a single file rather than a directory (unlike the
    docker launcher's --input_dir, this doesn't batch multiple jobs).

    execute defaults to False: prints the command(s) instead of running
    them, since AF3 runs are expensive and this is easy to fire off by
    accident inside a loop. Pass --execute once you've checked the printed
    command(s).
    """
    json_paths = sorted(Path(json_dir).glob("*.json"))
    if not json_paths:
        print(f"No json files found in {json_dir}, nothing to run.")
        return []

    commands = []
    for json_path in json_paths:
        cmd = (
            f"conda run -p {conda_env_path} python {af3_script_path} "
            f"--json_path={json_path} "
            f"--output_dir={output_dir} "
            f"--db_dir={db_dir} "
            f"--model_dir={model_dir} "
            f"--num_diffusion_samples={num_diffusion_samples}"
        )
        commands.append(cmd)
        print("Command:", cmd)
        if execute:
            subprocess.run(cmd, shell=True, check=True)

    if not execute:
        print(f"(execute=False, not running -- pass --execute to launch, "
              f"{len(commands)} command(s) above)")
    return commands


### INPUTS #################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AF3 json inputs with templates, then run AF3."
    )
    parser.add_argument(
        "--fasta-path", required=True,
        help="Fasta with the '>file_ID|chain_id|template_cif_path|"
             "template_chain_id' header convention (see module docstring).",
    )
    parser.add_argument(
        "--json-output-dir", default="af3_inputs",
        help="Where to write the generated AF3 input jsons. (default: %(default)s)",
    )
    parser.add_argument(
        "--af-output-dir", default="af3_outputs",
        help="AF3's own --output_dir. (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the AF3 model parameters directory.",
    )
    parser.add_argument(
        "--db-dir", required=True,
        help="Path to the AF3 genetic/public databases directory.",
    )
    parser.add_argument(
        "--json-schema-version", type=int, default=1, choices=[1, 2, 3, 4],
        help="AF3 input json 'version' field. Controls whether templates "
             "embed mmcif text (v1) or reference mmcifPath (v2+) -- must "
             "match what your installed AF3 build accepts, not just what "
             "the current upstream docs describe. Default 1, confirmed "
             "against this workstation's actual install. (default: %(default)s)",
    )
    parser.add_argument(
        "--conda-env-path", default="/home/ingemar/anaconda3/envs/alphafold3",
        help="Path (-p) to the conda env AF3 runs in. (default: %(default)s)",
    )
    parser.add_argument(
        "--af3-script-path", default="/mnt/data/alphafold3/run_alphafold.py",
        help="Path to run_alphafold.py. (default: %(default)s)",
    )
    parser.add_argument(
        "--num-diffusion-samples", type=int, default=1,
        help="AF3's --num_diffusion_samples. (default: %(default)s)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually launch AF3. Without this flag, the command(s) are "
             "printed but not run.",
    )
    return parser.parse_args()


def parse_fasta_to_df(fasta_path) -> pd.DataFrame:
    """
    Parse the '>file_ID|chain_id|template_cif_path|template_chain_id' header
    convention described at the top of this file into a dataframe, one row
    per entity/chain.
    """
    records = []
    header, seq_lines = None, []

    def flush():
        if header is None:
            return
        parts = [p.strip() for p in header.split("|")]
        file_id, chain_id = parts[0], parts[1]
        template_cif_path = parts[2] if len(parts) > 2 else ""
        template_chain_id = parts[3] if len(parts) > 3 else ""
        # Optional: path to the original, unextracted PDB cif this template
        # chain came from -- only needed if template_cif_path is missing
        # release-date metadata (common after single-chain extraction).
        # See graft_release_date_metadata.
        template_original_cif_path = parts[4] if len(parts) > 4 else ""
        records.append({
            "file_ID": file_id,
            "chain_id": chain_id,
            "sequence": "".join(seq_lines),
            "template_cif_path": template_cif_path,
            "template_chain_id": template_chain_id,
            "template_original_cif_path": template_original_cif_path,
        })

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        flush()

    return pd.DataFrame(records)


### MAIN ###################################################
if __name__ == "__main__":
    args = parse_args()
    Path(args.json_output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Transform fasta to dataframe
    df = parse_fasta_to_df(args.fasta_path)

    # 2. Generate AF3 json file per job (grouped by file_ID). Inside, each
    #    entity's template gets its own sequence_extractor + aligner run.
    for file_id, group in df.groupby("file_ID", sort=False):
        rows = [row for _, row in group.iterrows()]
        n = len(rows)
        if n == 1:
            generate_templated_AF_json_monomer(rows[0], args.json_output_dir, args.json_schema_version)
        elif n == 2:
            generate_templated_AF_json_protomer(rows, args.json_output_dir, args.json_schema_version)
        elif n == 4:
            generate_templated_AF_json_dimer(rows, args.json_output_dir, args.json_schema_version)
        else:
            raise ValueError(
                f"{file_id}: got {n} chains, only 1/2/4 (monomer/protomer/"
                f"dimer) are wired up -- add another wrapper if you need "
                f"a different stoichiometry."
            )

    # 3. Run AF3 (prints the command(s); pass --execute to actually launch)
    run_af(
        args.json_output_dir,
        args.af_output_dir,
        args.model_dir,
        args.db_dir,
        conda_env_path=args.conda_env_path,
        af3_script_path=args.af3_script_path,
        num_diffusion_samples=args.num_diffusion_samples,
        execute=args.execute,
    )