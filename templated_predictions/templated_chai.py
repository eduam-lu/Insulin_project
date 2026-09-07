"""
Script to run template-based predictions with Chai-1.

Chai's own fasta parser (chai_lab.data.dataset.inference_dataset.read_inputs)
only accepts headers of the exact form '>protein|name=chainA' -- adding
extra '|'-delimited fields to smuggle in template info raises a hard
ValueError (checked against the actual installed chai_lab source, not
assumed). So unlike the AF3 script, template info here lives in a small
separate manifest CSV instead of the fasta header:

    chain_id,template_cif_path,template_chain_id
    chainA,templates/mytpl1.cif,A
    chainB,templates/mytpl2.cif,A

- chain_id must match the 'name=' value in the fasta exactly.
- One cif file per template (per your note) -- not the "one shared cif,
  multiple chains" pattern from the dimer-templating discussion. Each row
  gets its own m8 line and its own staged cif.
- A chain with no row in the manifest just runs untemplated.

No aligner/sequence_extractor needed here (that was an AF3-specific step):
Chai aligns query-to-template itself internally via kalign, so all this
script has to do is point it at the right region -- passing query_start=1,
query_end=len(sequence) always requests "the whole thing", per Chai's own
slicing semantics (see build_m8_for_chains below).

The number of chains in the fasta (1/2/4/...) falls out automatically from
looping over feature_context.chains -- no separate monomer/protomer/tetramer
functions needed like in the AF3 script, since Chai's per-chain templating
doesn't change shape with chain count the way an AF3 json's structure does.
"""

### IMPORTS ###################################################
import argparse
import gzip
import shutil
from pathlib import Path

import pandas as pd
from chai_lab.chai1 import (
    read_inputs,
    load_chains_from_raw,
    make_all_atom_feature_context,
    run_folding_on_context,
)
from chai_lab.data.dataset.templates.context import get_template_context
from chai_lab.data.parsing.structure.entity_type import EntityType

### FUNCTIONS #################################################


def load_template_manifest(manifest_path) -> pd.DataFrame:
    """chain_id,template_cif_path,template_chain_id -- indexed by chain_id."""
    df = pd.read_csv(manifest_path, dtype=str)
    required = {"chain_id", "template_cif_path", "template_chain_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")
    return df.set_index("chain_id")


def stage_template_cif(source_path, cif_cache: Path, identifier: str) -> Path:
    """
    Copy/compress source_path into cif_cache/{identifier}.cif.gz -- the exact
    name+location chai_lab's downloader checks before hitting RCSB, so this
    is what makes it use your structure instead of trying to download one.
    Re-stages on every call (not skip-if-exists) so edits to your source cif
    are picked up on the next run instead of silently using a stale copy.
    """
    source_path = Path(source_path)
    dest_path = cif_cache / f"{identifier}.cif.gz"
    if source_path.suffix == ".gz":
        shutil.copy(source_path, dest_path)
    else:
        with open(source_path, "rb") as f_in, gzip.open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return dest_path


def _sanitize_identifier(cif_path) -> str:
    """Turn a cif filename into a usable, unique-enough m8/cache identifier."""
    stem = Path(cif_path).stem
    if stem.endswith(".cif"):  # strip a second suffix, e.g. foo.cif.gz -> foo
        stem = Path(stem).stem
    return "".join(c for c in stem.upper() if c.isalnum()) or "TPL"


def build_m8_for_chains(chains, manifest: pd.DataFrame, cif_cache: Path, m8_path: Path) -> int:
    """
    Analyse the (already-parsed) fasta chains against the manifest, stage
    each referenced cif, and write one m8 row per templated protein chain.
    Returns the number of rows written -- 0 means nothing to template.
    """
    rows = []
    for chain in chains:
        if chain.entity_data.entity_type != EntityType.PROTEIN:
            continue
        chain_id = chain.entity_data.entity_name
        if chain_id not in manifest.index:
            continue

        record = manifest.loc[chain_id]
        identifier = _sanitize_identifier(record["template_cif_path"])
        stage_template_cif(record["template_cif_path"], cif_cache, identifier)

        query_len = len(chain.entity_data.sequence)
        # subject_end deliberately oversized: chai slices the template's
        # resolved sequence as seq[start:end], and Python silently clips an
        # out-of-range end to the actual length -- so this always requests
        # "the whole resolved template", without us having to pre-parse the
        # cif just to find its length.
        rows.append(
            "\t".join([
                chain_id,
                f"{identifier}_{record['template_chain_id']}",
                "1.0", str(query_len), "0", "0",
                "1", str(query_len),
                "1", "999999",
                "0.0", "999", f"{query_len}M",
            ])
        )

    m8_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)


def run_templated_chai_prediction(
    fasta_file: Path,
    output_dir: Path,
    cif_cache: Path,
    manifest_path: Path,
    use_msa_server: bool = True,
    msa_directory: Path | None = None,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_cache.mkdir(parents=True, exist_ok=True)

    # 1. Parse the fasta the same way chai itself does, to get real Chain
    #    objects -- this is the "analyse my fasta programmatically" step:
    #    however many protein chains come out is however many m8 rows (or
    #    fewer, for any chain absent from the manifest) get written below,
    #    monomer/protomer/tetramer/whatever, no branching needed.
    chains = load_chains_from_raw(read_inputs(fasta_file, length_limit=None))
    n_protein = sum(1 for c in chains if c.entity_data.entity_type == EntityType.PROTEIN)
    print(f"{fasta_file}: {n_protein} protein chain(s) "
          f"({ {1: 'monomer', 2: 'protomer', 4: 'tetramer'}.get(n_protein, f'{n_protein}-mer') })")

    # 2. Build the manifest-driven m8 file + stage each referenced cif
    manifest = load_template_manifest(manifest_path)
    m8_path = output_dir / "custom.m8"
    n_templated = build_m8_for_chains(chains, manifest, cif_cache, m8_path)
    print(f"Templated {n_templated}/{n_protein} protein chain(s), see {m8_path}")

    # 3. Build everything else (MSA/embeddings/structure) normally, no
    #    templates yet -- same as the manual AF3-style approach we used
    #    before, since run_inference's own output_dir must start empty and
    #    can't accommodate our pre-staged cif_cache being reused across runs.
    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_file,
        output_dir=output_dir,
        use_esm_embeddings=True,
        use_msa_server=use_msa_server,
        msa_directory=msa_directory,
        templates_path=None,
    )

    # 4. Build the template context ourselves, pointing at our staged cifs
    if n_templated > 0:
        template_context = get_template_context(
            chains=feature_context.chains,
            template_hits_m8=m8_path,
            template_cif_cache_folder=cif_cache,
        )
        feature_context.template_context = template_context

    # 5. Run
    return run_folding_on_context(
        feature_context,
        output_dir=output_dir,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        low_memory=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Chai-1 template-based prediction."
    )
    parser.add_argument("--fasta-path", required=True, type=Path,
                         help="Chai-format fasta, e.g. '>protein|name=chainA' headers.")
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--cif-cache-dir", default=Path("my_templates"), type=Path,
                         help="Where template cifs get staged for chai_lab to find.")
    parser.add_argument("--template-manifest", required=True, type=Path,
                         help="CSV: chain_id,template_cif_path,template_chain_id")
    msa_group = parser.add_mutually_exclusive_group()
    msa_group.add_argument("--use-msa-server", action="store_true", default=True,
                            help="Use the ColabFold MSA server (default).")
    msa_group.add_argument("--msa-directory", type=Path,
                            help="Use precomputed .aligned.pqt MSAs from this directory instead.")
    parser.add_argument("--num-trunk-recycles", type=int, default=3)
    parser.add_argument("--num-diffn-timesteps", type=int, default=200)
    parser.add_argument("--num-diffn-samples", type=int, default=5)
    return parser.parse_args()


### INPUTS #######################################################

### MAIN #########################################################
if __name__ == "__main__":
    args = parse_args()
    use_msa_server = args.use_msa_server and args.msa_directory is None

    run_templated_chai_prediction(
        fasta_file=args.fasta_path,
        output_dir=args.output_dir,
        cif_cache=args.cif_cache_dir,
        manifest_path=args.template_manifest,
        use_msa_server=use_msa_server,
        msa_directory=args.msa_directory,
        num_trunk_recycles=args.num_trunk_recycles,
        num_diffn_timesteps=args.num_diffn_timesteps,
        num_diffn_samples=args.num_diffn_samples,
    )