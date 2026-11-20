import csv
from pathlib import Path

import numpy as np

from read_output import (
    GRADIENT_METHODS,
    OK,
    accuracy_reference_dir,
    basis_dir,
    collect_gradient_data,
    describe_system,
    format_and_list,
    format_sigfigs,
    get_gradient_matrix,
    get_status,
    get_valid_runs,
    latex_tables,
    status_sentence,
)

OUTDIR = Path(__file__).parent / "accuracy_output"
OUTDIR.mkdir(exist_ok=True)

# Approximate methods are kept out of the tables: RI and Grid-K exchange introduce their own error.
ACCURACY_METHODS = {code: method for code, method in GRADIENT_METHODS.items() if code not in ("exess_ri", "cuest_gridk")}


def get_method_rmsd(row, molecule, basis, code):
    runs = get_valid_runs(row, code)
    if runs is None:
        return np.nan
    reference = accuracy_reference_dir(code, basis) / basis_dir(basis) / f"{molecule}{row['size']}.out"
    difference = get_gradient_matrix("gpu4pyscf", reference) - get_gradient_matrix(code, runs[0])
    return float(np.sqrt(np.mean(difference**2)))


def csv_rmsd(rows):
    with open(OUTDIR / "gradient_rmsd.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["basis", "molecule", "size", *ACCURACY_METHODS])
        for (basis, molecule), panel_rows in rows.items():
            for row in panel_rows:
                cells = []
                for code in ACCURACY_METHODS:
                    status = get_status(panel_rows, row, code)
                    cells.append(f"{get_method_rmsd(row, molecule, basis, code):.6e}" if status == OK else status or "")
                writer.writerow([basis, molecule, row["size"], *cells])


def print_mean_rmsd(rows):
    for code, method in ACCURACY_METHODS.items():
        rmsds = [
            get_method_rmsd(row, molecule, basis, code)
            for (basis, molecule), panel_rows in rows.items()
            for row in panel_rows
        ]
        print(f"{method['label']}: {np.nanmean(rmsds):.3e} Eh/Bohr ({np.count_nonzero(~np.isnan(rmsds))} systems)")


def print_gridk_rmsd(rows):
    by_basis = {}
    for (basis, molecule), panel_rows in rows.items():
        for row in panel_rows:
            by_basis.setdefault(basis, []).append(get_method_rmsd(row, molecule, basis, "cuest_gridk"))
    means = {basis: np.nanmean(rmsds) for basis, rmsds in by_basis.items()}
    overall = np.nanmean([rmsd for rmsds in by_basis.values() for rmsd in rmsds])
    print(f"cuEST (Grid-K): {overall:.3e} Eh/Bohr mean; by basis: " + ", ".join(f"{b} {m:.2e}" for b, m in means.items()))


def latex_rmsd_tables(rows):
    def caption(molecule, blocks, markers):
        all_bases = [basis for bases, _, _ in blocks for basis in bases]
        reference = (
            r"relative to tight-threshold \texttt{GPU4PySCF} reference gradients "
            r"(Cartesian for 6-31G* and 6-31G**, spherical for the others; "
            r"\texttt{QUICK} and \texttt{VeloxChem} use Cartesian and spherical references "
            r"respectively throughout)."
        )
        sentences = [
            f"HF gradient RMSD ($10^{{-6}}$ $E_h$/Bohr) for {describe_system(molecule, blocks)} "
            f"with the {format_and_list(all_bases)} basis sets, {reference}",
            status_sentence(markers),
        ]
        return " ".join(filter(None, sentences))

    tex = latex_tables(
        rows,
        ACCURACY_METHODS,
        cell_fn=lambda row, molecule, basis, code: format_sigfigs(1e6 * get_method_rmsd(row, molecule, basis, code)),
        caption_fn=caption,
        label_prefix="rmsd",
    )
    (OUTDIR / "gradient_rmsd_tables.tex").write_text(tex)


def main():
    data = collect_gradient_data()
    csv_rmsd(data)
    latex_rmsd_tables(data)
    print_mean_rmsd(data)
    print_gridk_rmsd(data)


if __name__ == "__main__":
    main()
