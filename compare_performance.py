import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib import ticker

from read_output import (
    GRADIENT_METHODS,
    OK,
    OUTPUT_ROOT,
    basis_dir,
    code_label_tex,
    collect_gradient_data,
    describe_system,
    format_and_list,
    format_sigfigs,
    get_gradient_time,
    get_status,
    get_valid_runs,
    latex_tables,
    run_status,
    run_time,
    status_cell,
    status_sentence,
)

plt.style.use("science")
plt.rcParams["text.latex.preamble"] += r"\usepackage[version=4]{mhchem}"

OUTDIR = Path(__file__).parent / "performance_output"
OUTDIR.mkdir(exist_ok=True)

FIGSIZE = (6, 3)
FONTSIZE = 12

MISC_SYSTEMS = ["taxol", "valinomycin", "olestra", "crambin", "ubiquitin"]
MISC_BASIS = "def2-SVP"
LATEX_CODE_ORDER = ["exess", "terachem", "gpu4pyscf", "quick", "veloxchem", "cuest_direct", "cuest_gridk"]
MISC_CODES = ["exess", "terachem", "gpu4pyscf", "cuest_direct", "cuest_gridk", "quick", "veloxchem"]
LEGEND_COLUMNS = [
    ["TeraChem", "QUICK", "EXESS (RI)", "EXESS"],
    ["VeloxChem", "cuEST (Grid-K)", "cuEST"],
    ["GPU4PySCF"],
]
PLOT_STYLES = {
    "terachem": {"color": "purple", "marker": "s", "linestyle": (0, (1, 1))},
    "veloxchem": {"color": "deeppink", "marker": "v", "linestyle": "-."},
    "quick": {"color": "green", "marker": "D", "linestyle": (0, (3, 3, 1, 3, 1, 3))},
    "gpu4pyscf": {"color": "red", "marker": "o", "linestyle": (0, (4, 2, 4, 2, 1, 2))},
    "exess": {"color": "blue", "marker": "^", "linestyle": ":"},
    "exess_ri": {"color": "grey", "marker": "*", "linestyle": "-"},
    "cuest_direct": {"color": "teal", "marker": "P", "linestyle": (0, (5, 1))},
    "cuest_gridk": {"color": "darkorange", "marker": "X", "linestyle": (0, (3, 1, 1, 1, 1, 1))},
}
EXCHANGE_TIME_PATTERNS = {
    "exess": r"Exchange \(K\) gradient:\s+([\d.]+)",
    "terachem": r"^\s*Gradient:\s+K\s+elapsed time:\s+([\d.]+)\s*sec",
}
TOTAL_TIME_PATTERNS = {
    "terachem": r"^\| Total processing time:\s+([\d.]+)\s*sec",
    "veloxchem": r"^Total processing time\s+([\d.]+)\s+sec",
    "quick": r"\|\s*TOTAL TIME\s*=\s*([\d.]+)",
    "gpu4pyscf": r"^Total processing time\s+([\d.]+)\s+sec",
    "exess_ri": r"Total processing time ([\d.]+) seconds",
    "cuest_direct": r"^Total processing time\s+([\d.]+)\s+sec",
    "cuest_gridk": r"^Total processing time\s+([\d.]+)\s+sec",
}
SIZE_TICKS = {
    "water": {"label": "\\ce{{(H2O){:.0f}}}", "fontsize": FONTSIZE},
    "gly": {"label": "gly\\textsubscript{{{:.0f}}}", "fontsize": FONTSIZE + 1},
}

# (slope, intercept) of the basis-function count in the system size, in each basis's own AO convention.
NUM_BASIS_FUNCS = {
    "water": {
        "cc-pVDZ": (24, 0),
        "def2-SVP": (24, 0),
        "6-31G*": (19, 0),
        "6-31G**": (25, 0),
        "cc-pVTZ": (58, 0),
        "def2-TZVP": (43, 0),
    },
    "gly": {
        "cc-pVDZ": (71, 24),
        "def2-SVP": (71, 24),
        "6-31G*": (66, 19),
        "6-31G**": (75, 25),
        "cc-pVTZ": (162, 58),
        "def2-TZVP": (142, 43),
    },
}


def set_pretty_axes(ax, sizes, basis, molecule):
    slope, intercept = NUM_BASIS_FUNCS[molecule][basis]
    ax.set_xlabel(f"Number of basis functions ({basis})", fontsize=FONTSIZE)
    ax.xaxis.set_minor_locator(ticker.FixedLocator([]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val)}"))
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=FONTSIZE)

    secax = ax.secondary_xaxis("top", functions=(lambda x: (x - intercept) / slope, lambda x: x * slope + intercept))
    secax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: SIZE_TICKS[molecule]["label"].format(val)))
    secax.xaxis.set_major_locator(ticker.FixedLocator(sizes[1::2]))
    secax.xaxis.set_minor_locator(ticker.FixedLocator(sizes))
    secax.tick_params(axis="x", which="both", bottom=False, top=True, labelsize=SIZE_TICKS[molecule]["fontsize"])

    ax.set_ylabel("Gradient time (s)", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(False)
    ax.tick_params(axis="y", which="minor", left=True, right=True)


def add_legend_headroom(fig, ax, legend):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    entry_boxes = []
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        if not text.get_text():
            continue
        handle_box, text_box = handle.get_window_extent(renderer), text.get_window_extent(renderer)
        entry_boxes.append((min(handle_box.x0, text_box.x0), max(handle_box.x1, text_box.x1), text_box.y0))

    y_min, y_max = ax.get_ylim()
    span = np.log10(y_max / y_min)
    for line in ax.get_lines():
        xdata, ydata = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        finite = np.isfinite(ydata)
        display_x = ax.transData.transform(np.column_stack([xdata[finite], ydata[finite]]))[:, 0]
        for x, y in zip(display_x, ydata[finite]):
            ceilings = [y0 for x0, x1, y0 in entry_boxes if x0 <= x <= x1]
            if ceilings:
                target = (min(ceilings) - 0.06 * ax_box.height - ax_box.y0) / ax_box.height
                span = max(span, np.log10(y / y_min) / min(max(target, 0.2), 0.95))

    ax.set_ylim(y_min, y_min * 10**span)


def ordered_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    columns = [[label for label in column if label in by_label] for column in LEGEND_COLUMNS]
    height = max(map(len, columns))
    blank = plt.Line2D([], [], linestyle="None", color="none")
    padded = [column + [""] * (height - len(column)) for column in columns]
    labels = [label for column in padded for label in column]
    return ax.legend(
        [by_label.get(label, blank) for label in labels],
        labels,
        fontsize=FONTSIZE - 2,
        loc="upper left",
        ncol=len(columns),
        columnspacing=0.6,
        handlelength=1.2,
        handletextpad=0.3,
        borderaxespad=0.3,
    )


def plot_power_law_fit(ax, xvals, times, color):
    # The three smallest systems are dominated by setup overhead and would bias the exponent low.
    x, y = np.array(xvals, dtype=float), np.array(times)
    finite = np.isfinite(y)
    x, y = x[finite][3:], y[finite][3:]
    if len(x) < 2:
        return None
    exponent, log_prefactor = np.polyfit(np.log(x), np.log(y), 1)
    xs_fit = np.linspace(x.min(), x.max(), 100)
    ys_fit = np.exp(log_prefactor) * xs_fit**exponent
    ax.plot(xs_fit, ys_fit, color=color, linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    return ax.annotate(
        f"$N^{{{exponent:.2f}}}$",
        xy=(xs_fit[-1], ys_fit[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        fontsize=FONTSIZE - 2,
        va="center",
        ha="left",
        annotation_clip=False,
    )


def separate_exponent_labels(fig, ax, labels):
    right = max(label.xy[0] for label in labels)
    labels = [label for label in labels if label.xy[0] == right]
    points = 72 / fig.dpi
    ys = sorted((ax.transData.transform(label.xy)[1] * points + label.xyann[1], label) for label in labels)
    pos = [y for y, _ in ys]
    min_gap = 1.3 * (FONTSIZE - 2)
    for _ in range(100):
        for i in range(len(pos) - 1):
            overlap = min_gap - (pos[i + 1] - pos[i])
            if overlap > 0:
                pos[i] -= overlap / 2
                pos[i + 1] += overlap / 2
    for (y, label), new in zip(ys, pos):
        label.xyann = (label.xyann[0], label.xyann[1] + new - y)


def clear_inner_labels(fig, ax, labels):
    right = max(label.xy[0] for label in labels)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        if label.xy[0] == right:
            continue
        others = [
            line.get_transform().transform_path(line.get_path())
            for line in ax.get_lines()
            if not line.get_label().startswith("_") and line.get_color() != label.get_color()
        ]
        for dy in sorted(range(-40, 41, 2), key=abs):
            label.xyann = (label.xyann[0], dy)
            box = label.get_window_extent(renderer).padded(fig.dpi / 72)
            if not any(path.intersects_bbox(box, filled=False) for path in others):
                break


def annotate_speedup(ax, slow, fast, xvals):
    best = np.nanargmax(np.array(slow) / np.array(fast))
    x_range = max(xvals) - min(xvals)
    x = xvals[best] - 0.02 * x_range
    bar = 0.01 * x_range
    ax.plot([x, x], [slow[best], fast[best]], color="black", linestyle="-")
    ax.hlines(slow[best], x - bar, x + bar, color="black")
    ax.hlines(fast[best], x - bar, x + bar, color="black")
    last = best == len(xvals) - 1
    ax.text(
        x + (-0.016 if last else 0.016) * x_range,
        math.sqrt(slow[best] * fast[best]),
        f"{slow[best] / fast[best]:.1f}$\\times$",
        fontsize=FONTSIZE - 1,
        va="center",
        ha="right" if last else "left",
    )


def plot_all(rows):
    for (basis, molecule), panel_rows in rows.items():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        slope, intercept = NUM_BASIS_FUNCS[molecule][basis]
        sizes = [row["size"] for row in panel_rows]
        xvals = [size * slope + intercept for size in sizes]
        times = {}
        labels = {}
        for code, style in PLOT_STYLES.items():
            times[code] = [
                get_gradient_time(code, row[code]) if get_valid_runs(row, code) else np.nan for row in panel_rows
            ]
            if all(np.isnan(times[code])):
                continue
            ax.plot(
                xvals,
                times[code],
                label=GRADIENT_METHODS[code]["label"],
                fillstyle="none",
                **style,
            )
            labels[code] = plot_power_law_fit(ax, xvals, times[code], style["color"])
        slow = np.fmin.reduce([t for code, t in times.items() if code != "exess"])
        annotate_speedup(ax, slow, times["exess"], xvals)

        set_pretty_axes(ax, sizes, basis, molecule)
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + 0.08 * (right - left))
        legend = ordered_legend(ax)
        add_legend_headroom(fig, ax, legend)
        separate_exponent_labels(fig, ax, [label for label in labels.values() if label])
        clear_inner_labels(fig, ax, [label for label in labels.values() if label])
        fig.savefig(OUTDIR / f"gradient_time_{molecule}_{basis_dir(basis)}.png", dpi=600)
        plt.close(fig)


def time_cell(row, code, status):
    return f"{get_gradient_time(code, row[code]):.3f}" if status == OK else status or ""


def csv_times(rows):
    with open(OUTDIR / "gradient_timings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["basis", "molecule", "size", *GRADIENT_METHODS])
        for (basis, molecule), panel_rows in rows.items():
            for row in panel_rows:
                cells = [time_cell(row, code, get_status(panel_rows, row, code)) for code in GRADIENT_METHODS]
                writer.writerow([basis, molecule, row["size"], *cells])


def gradient_percent(code, runs):
    totals = {
        run: float(m.group(1)) for run in runs if (m := re.search(TOTAL_TIME_PATTERNS[code], run.read_text(), re.M))
    }
    run = min(totals, key=totals.get)
    return 100 * run_time(code, run) / totals[run]


def print_average_percents(rows):
    for code in TOTAL_TIME_PATTERNS:
        percents = [
            gradient_percent(code, row[code])
            for panel_rows in rows.values()
            for row in panel_rows
            if get_valid_runs(row, code)
        ]
        note = " (non-incremental Fock build)" if code == "cuest_gridk" else ""
        print(f"{GRADIENT_METHODS[code]['label']}{note} gradient share of total runtime: {np.mean(percents):.1f}%")


def threshold_sentence(codes):
    # One group per threshold, in order of first appearance: "$10^{-11}$ for A, B and C; ...".
    groups = {}
    for code in codes:
        label = code_label_tex(GRADIENT_METHODS[code]["label"]).replace(" (", " ").rstrip(")")
        groups.setdefault(GRADIENT_METHODS[code]["threshold"], []).append(label)
    parts = [f"{threshold} for {format_and_list(labels)}" for threshold, labels in groups.items()]
    joined = parts[0] if len(parts) == 1 else "; ".join(parts[:-1]) + "; and " + parts[-1]
    return f"The screening thresholds were {joined}."


def latex_timing_tables(rows):
    def caption(molecule, blocks, markers):
        all_bases = [basis for bases, _, _ in blocks for basis in bases]
        ordered = [code for code in LATEX_CODE_ORDER if any(code in codes for _, _, codes in blocks)]
        sentences = [
            f"HF gradient build timings (seconds) for {describe_system(molecule, blocks)} "
            f"with the {format_and_list(all_bases)} basis sets.",
            "Timings were repeated three times on an NVIDIA H200 GPU and the minimum reported.",
            threshold_sentence(ordered),
            status_sentence(markers),
        ]
        return " ".join(filter(None, sentences))

    tex = latex_tables(
        rows,
        GRADIENT_METHODS,
        cell_fn=lambda row, molecule, basis, code: format_sigfigs(get_gradient_time(code, row[code])),
        caption_fn=caption,
        label_prefix="timings",
    )
    (OUTDIR / "gradient_timings_tables.tex").write_text(tex)


def collect_misc_data():
    rows = []
    for system in MISC_SYSTEMS:
        row = {"system": system}
        for code in GRADIENT_METHODS:
            run = OUTPUT_ROOT / code / "misc" / f"{system}.out"
            if run.exists():
                row[code] = [run]
        rows.append(row)
    return rows


def csv_misc_times(rows):
    with open(OUTDIR / "misc_gradient_timings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        codes = [code for code in GRADIENT_METHODS if code != "exess_ri"]
        writer.writerow(["basis", "system", *codes])
        for row in rows:
            cells = [time_cell(row, code, run_status(code, row[code]) if code in row else None) for code in codes]
            writer.writerow([MISC_BASIS, row["system"], *cells])


def thousands(text):
    integer, dot, fraction = text.partition(".")
    if len(integer) > 3:
        integer = f"{integer[:-3]}\\,{integer[-3:]}"
    return integer + dot + fraction


def latex_misc_table(rows):
    markers = set()
    lines = []
    for row in rows:
        num_basis_funcs = re.search(r"^number of NR cGTOs\s*=\s*(\d+)", row["gpu4pyscf"][0].read_text(), re.MULTILINE)
        cells = [row["system"].ljust(11), thousands(num_basis_funcs.group(1))]
        for code in MISC_CODES:
            status = run_status(code, row[code]) if code in row else None
            if status == OK:
                cells.append(thousands(format_sigfigs(get_gradient_time(code, row[code]))))
            else:
                markers.add(status)
                cells.append(status_cell(status))
        lines.append(" & ".join(cells) + r" \\")

    header = ["", r"$N_{\mathrm{bf}}$"] + [code_label_tex(GRADIENT_METHODS[code]["label"]) for code in MISC_CODES]
    caption = (
        "Wall time (s) for a single HF nuclear gradient build on one NVIDIA H200 GPU for "
        f"{len(rows)} large molecules with the {MISC_BASIS} basis set. " + status_sentence(markers)
    )
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:large-molecules}",
        r"\small",
        r"\begin{tabular}{lr" + "r" * len(MISC_CODES) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (OUTDIR / "misc_gradient_timings_table.tex").write_text("\n".join(tex) + "\n")


def exchange_time(code, run):
    return sum(map(float, re.findall(EXCHANGE_TIME_PATTERNS[code], run.read_text(), re.MULTILINE)))


def fastest_run(code, runs):
    return min(runs, key=lambda run: run_time(code, run))


def print_exchange_fractions(rows):
    shares, speedup_shares = [], []
    for (basis, molecule), panel_rows in rows.items():
        row = next(
            (r for r in reversed(panel_rows) if get_valid_runs(r, "exess") and get_valid_runs(r, "terachem")), None
        )
        if row is None:
            continue
        ex_run, tc_run = fastest_run("exess", row["exess"]), fastest_run("terachem", row["terachem"])
        ex_total, tc_total = run_time("exess", ex_run), run_time("terachem", tc_run)
        ex_k, tc_k = exchange_time("exess", ex_run), exchange_time("terachem", tc_run)
        shares.append(100 * ex_k / ex_total)
        speedup_shares.append(100 * (tc_k - ex_k) / (tc_total - ex_total))
    print(f"EXESS K share of gradient time: {np.mean(shares):.1f}%")
    print(f"K share of EXESS speedup over TeraChem: {np.mean(speedup_shares):.1f}%")


def print_average_speedups(rows):
    for code in ("terachem", "gpu4pyscf", "quick", "veloxchem", "cuest_direct", "cuest_gridk"):
        speedups = [
            get_gradient_time(code, row[code]) / get_gradient_time("exess", row["exess"])
            for panel_rows in rows.values()
            for row in panel_rows
            if get_valid_runs(row, code) and get_valid_runs(row, "exess")
        ]
        geometric = np.exp(np.mean(np.log(speedups)))
        print(
            f"EXESS speedup over {GRADIENT_METHODS[code]['label']}: "
            f"{np.mean(speedups):.1f}x arithmetic mean, {geometric:.1f}x geometric mean, {min(speedups):.1f}x minimum"
        )


def main():
    data = collect_gradient_data()
    csv_times(data)
    latex_timing_tables(data)
    plot_all(data)
    print_exchange_fractions(data)
    print_average_speedups(data)
    print_average_percents(data)

    misc = collect_misc_data()
    csv_misc_times(misc)
    latex_misc_table(misc)


if __name__ == "__main__":
    main()
