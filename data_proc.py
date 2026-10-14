import json
from pathlib import Path


def export_results_to_tsv(json_path, out_path=None, print_output=True):
    """
    Read result JSON and export in a tab-separated format suitable
    for copy–paste into Google Sheets.

    json_path: path to main_exp_*.json
    out_path: optional path to .tsv file
    print_output: if True, print to stdout
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        data = json.load(f)

    # Header matching your example
    header = [
        "Task Name",
        "Depth Estimator",
        "Camera Setting",   # we’ll join the camera_names list
        "Resolution",
        "Input",            # from mask_input (True/False)
        "ICP n_init",       # not in JSON -> leave blank
        "ICP max_iter",     # not in JSON -> leave blank
        "SR",
        "SR std",
    ]

    lines = ["\t".join(header)]

    for entry in data:
        task_name       = entry.get("task_name", "")
        depth_estimator = entry.get("depth_estimator", "")
        camera_names    = entry.get("camera_names", [])
        resolution      = entry.get("resolution", "")
        mask_input      = entry.get("mask_input", None)
        avg_sr          = entry.get("avg_sr", "")
        std_sr          = entry.get("std_sr", "")

        # Join camera names like "front+left_shoulder"
        camera_setting = " / ".join(camera_names) if camera_names else ""

        # Map mask_input to something human-readable
        if mask_input is True:
            input_str = "masked RGB"
        elif mask_input is False:
            input_str = "RGB"
        else:
            input_str = "None"

        # Format floats nicely for Sheets
        sr_str = f"{avg_sr:.3f}" if isinstance(avg_sr, (int, float)) else ""
        std_str = f"{std_sr:.3f}" if isinstance(std_sr, (int, float)) else ""

        row = [
            task_name,
            depth_estimator,
            camera_setting,
            str(resolution),
            input_str,
            "",        # ICP n_init (not in JSON)
            "",        # ICP max_iter (not in JSON)
            sr_str,
            std_str,
        ]
        lines.append("\t".join(row))

    tsv_text = "\n".join(lines)

    if print_output:
        print(tsv_text)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.write_text(tsv_text)
        print(f"Saved TSV to: {out_path}")

    return tsv_text


if __name__ == "__main__":
    export_results_to_tsv("results/rlbench/main_results_fixed_demo/vggt.json", 
                          out_path="results/rlbench/main_results_fixed_demo/vggt.tsv")