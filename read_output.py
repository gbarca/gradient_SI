import math
import re
from pathlib import Path

import numpy as np

OUTPUT_ROOT = Path(__file__).parent / "output"

OK, OOM, NC = "OK", "OOM", "NC"
STATUS_CAPTIONS = {
    OOM: "OOM denotes out of GPU memory",
    NC: "NC denotes the SCF did not converge",
}

TZ_BASES = ["cc-pVTZ", "def2-TZVP"]

GRADIENT_METHODS = {
    "terachem": {"label": "TeraChem", "kinds": ("cart", "sph"), "threshold": r"$10^{-11}$"},
    "veloxchem": {"label": "VeloxChem", "kinds": ("sph",), "threshold": r"$10^{-10}$"},
    "quick": {"label": "QUICK", "kinds": ("cart",), "threshold": r"$10^{-10}$"},
    "gpu4pyscf": {"label": "GPU4PySCF", "kinds": ("cart", "sph"), "threshold": r"$10^{-10.5}$"},
    "exess": {"label": "EXESS", "kinds": ("cart", "sph"), "threshold": r"$10^{-11}$"},
    "exess_ri": {"label": "EXESS (RI)", "kinds": ("cart", "sph"), "threshold": r"$10^{-11}$"},
    "cuest_direct": {"label": "cuEST", "kinds": ("cart", "sph"), "threshold": r"$10^{-11}$"},
    "cuest_gridk": {"label": "cuEST (Grid-K)", "kinds": ("cart", "sph"), "threshold": r"$10^{-11}$"},
}

MOLECULES = {
    "water": {"formula": "(H2O)", "caption_name": "globular water clusters"},
    "gly": {"formula": "gly", "caption_name": "linear glycine chains"},
}

GRADIENT_BASES = ["cc-pVDZ", "def2-SVP", "6-31G*", "6-31G**", *TZ_BASES]
SPHERICAL_BASES = {"cc-pVDZ", "def2-SVP", *TZ_BASES}
GRADIENT_TESTS = {
    "water": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
    "gly": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}
GRADIENT_TESTS_TZ = {
    "water": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "gly": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
}

GRADIENT_TIME_PATTERNS = {
    "gpu4pyscf": r"^Gradient wall time\s+([\d.]+)\s+sec",
    "veloxchem": r"^Gradient wall time\s+([\d.]+)\s+sec",
    "quick": r"\|\s*TOTAL GRADIENT TIME\s*=\s*([\d.]+)",
    "terachem": r"^\s*Gradient:\s+\S+\s+elapsed time:\s+([\d.]+)\s*sec",
    "exess": r"EXESS gradient wall time:\s+([\d.]+) s",
    "exess_ri": r"Total gradient time:\s+([\d.]+)s",
    "cuest_direct": r"^Gradient wall time\s+([\d.]+)\s+sec",
    "cuest_gridk": r"^Gradient wall time\s+([\d.]+)\s+sec",
}
NC_PATTERN = re.compile(r"NO CONVERGENCE|not converged", re.IGNORECASE)
XYZ_PATTERN = r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"


def basis_dir(basis):
    return basis.replace("*", "d")


def method_kind(code, basis):
    kinds = GRADIENT_METHODS[code]["kinds"]
    preferred = "sph" if basis in SPHERICAL_BASES else "cart"
    return preferred if preferred in kinds else kinds[0]


def accuracy_reference_dir(code, basis):
    return OUTPUT_ROOT / f"accurate_{method_kind(code, basis)}"


def collect_gradient_data():
    data = {}
    for molecule in GRADIENT_TESTS:
        for basis in GRADIENT_BASES:
            rows = []
            for size in (GRADIENT_TESTS_TZ if basis in TZ_BASES else GRADIENT_TESTS)[molecule]:
                row = {"size": size}
                for code in GRADIENT_METHODS:
                    runs = sorted((OUTPUT_ROOT / code / basis_dir(basis)).glob(f"{molecule}{size}_run*.out"))
                    if runs:
                        row[code] = runs
                rows.append(row)
            data[basis, molecule] = rows
    return data


def run_time(code, run):
    times = re.findall(GRADIENT_TIME_PATTERNS[code], run.read_text(), re.MULTILINE)
    return sum(map(float, times)) if times else np.nan


def get_gradient_time(code, runs):
    return min(run_time(code, run) for run in runs)


def _grad_gpu4pyscf(text):
    section = text.split("RHF gradients")[1].split("Gradient wall time")[0]
    return re.findall(r"^\s*\d+\s+\w+\s+" + XYZ_PATTERN, section, re.MULTILINE)


def _grad_after(header):
    return lambda text: re.findall(r"^\s*\d+\s+[A-Za-z]+\s+" + XYZ_PATTERN, text.split(header)[1], re.MULTILINE)


def _grad_terachem(text):
    section = text.split("dE/dZ")[1].split("---")[0]
    return re.findall(r"^\s*" + XYZ_PATTERN, section, re.MULTILINE)


def _grad_quick(text):
    section = text.split("ANALYTICAL GRADIENT:")[1].split("@ End Gradient Calculation")[0]
    return re.findall(r"^\s*\d+[XYZ]\s+[-\d.]+\s+([-\d.]+)", section, re.MULTILINE)


GRADIENT_MATRIX_PARSERS = {
    "gpu4pyscf": _grad_gpu4pyscf,
    "quick": _grad_quick,
    "exess": _grad_after("Nuclear gradient (Eh/bohr):"),
    "terachem": _grad_terachem,
    "veloxchem": _grad_after("Total nuclear gradient (Eh/bohr):"),
    "cuest_direct": _grad_after("Total nuclear gradient (Eh/bohr):"),
    "cuest_gridk": _grad_after("Total nuclear gradient (Eh/bohr):"),
}


def get_gradient_matrix(code, file):
    return np.array(GRADIENT_MATRIX_PARSERS[code](file.read_text()), dtype=float).reshape(-1, 3)


def run_status(code, runs):
    if NC_PATTERN.search(runs[0].read_text()):
        return NC
    if np.isnan(get_gradient_time(code, runs)):
        return OOM
    return OK


def get_status(panel_rows, row, code):
    if code in row:
        return run_status(code, row[code])
    # Sweeps stop at the first out-of-memory failure, so larger systems have no output at all.
    if any(r["size"] < row["size"] and code in r and run_status(code, r[code]) == OOM for r in panel_rows):
        return OOM
    return None


def get_valid_runs(row, code):
    if code in row and run_status(code, row[code]) == OK:
        return row[code]
    return None


def format_sigfigs(val):
    return f"{val:.{max(0, 2 - math.floor(math.log10(val)))}f}"


def format_and_list(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def code_label_tex(label):
    name, _, suffix = label.partition(" (")
    return f"\\texttt{{{name}}}" + (f" ({suffix}" if suffix else "")


def status_sentence(statuses):
    parts = [STATUS_CAPTIONS[s] for s in (OOM, NC) if s in statuses]
    return format_and_list(parts) + "." if parts else ""


def status_cell(status):
    return f"{{\\footnotesize {status}}}" if status else "-"


def describe_system(molecule, blocks):
    formula = MOLECULES[molecule]["formula"]
    ranges = [f"\\ce{{{formula}{sizes[0]}}}--\\ce{{{formula}{sizes[-1]}}}" for _, sizes, _ in blocks]
    if len(blocks) == 1:
        return f"{MOLECULES[molecule]['caption_name']} ({ranges[0]})"
    parts = [f"{rng} for the {name} basis sets" for rng, name in zip(ranges, ["double-$\\zeta$", "triple-$\\zeta$"])]
    parts[0] += ", top,"
    parts[1] += ", bottom"
    return f"{MOLECULES[molecule]['caption_name']} ({' and '.join(parts)})"


def latex_tables(rows, methods, cell_fn, caption_fn, label_prefix):
    standard_bases = [basis for basis in GRADIENT_BASES if basis not in TZ_BASES]
    tables = []
    for molecule in MOLECULES:
        blocks, all_statuses = [], set()
        for bases in (standard_bases, TZ_BASES):
            statuses = {
                (basis, code): [get_status(rows[basis, molecule], row, code) for row in rows[basis, molecule]]
                for basis in bases
                for code in methods
            }
            codes = [code for code in methods if any(any(statuses[basis, code]) for basis in bases)]
            sizes = [row["size"] for row in rows[bases[0], molecule]]
            blocks.append((bases, sizes, codes, statuses))
            all_statuses |= {status for cells in statuses.values() for status in cells} - {OK, None}

        ncols = len(blocks[0][1])
        header = f"\\ce{{{MOLECULES[molecule]['formula']}_{{$n$}}}}"
        lines = [r"\begin{table*}", "{", r"\begin{tabular}{ll" + "c" * ncols + "}"]
        for bases, sizes, codes, statuses in blocks:
            if bases is not standard_bases:
                lines.append(r"\hline")
            lines.append(f" & & \\multicolumn{{{ncols}}}{{c}}{{{header}, $n$ =}} \\\\ \\cline{{3-{ncols + 2}}}")
            lines.append(" & ".join(["Basis set & Code", *map(str, sizes)]) + r" \\ \hline")
            for basis in bases:
                for code in codes:
                    lead = f" \\multirow{{{len(codes)}}}{{*}}{{{basis}}}" if code == codes[0] else " "
                    cells = [
                        cell_fn(row, molecule, basis, code) if status == OK else status_cell(status)
                        for row, status in zip(rows[basis, molecule], statuses[basis, code])
                    ]
                    lines.append(" & ".join([lead, code_label_tex(methods[code]["label"]), *cells]) + r" \\")
                if basis != bases[-1]:
                    lines.append(r"\hline")

        caption_blocks = [(bases, sizes, codes) for bases, sizes, codes, _ in blocks]
        lines += [
            r"\end{tabular}",
            r"\caption{" + caption_fn(molecule, caption_blocks, all_statuses) + "}",
            r"\label{" + f"tab:{label_prefix}_{molecule}" + "}",
            "}",
            r"\end{table*}",
        ]
        tables.append("\n".join(lines))
    return "\n\n".join(tables) + "\n"
