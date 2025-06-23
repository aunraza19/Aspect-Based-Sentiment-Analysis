from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import TaskCodeOption
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset


def load_atepc_examples(dataset_name: str) -> list[str]:

    task = TaskCodeOption.Aspect_Polarity_Classification

    atepc_dataset_item = ATEPC.ATEPCDatasetList().__getattribute__(dataset_name)

    dataset_files = detect_infer_dataset(atepc_dataset_item, task)

    all_lines = []

    if isinstance(dataset_files, str):
        dataset_files = [dataset_files]

    for fpath in dataset_files:
        print(f"Loading ATEPC examples from: {fpath}")
        try:
            with open(fpath, "r", encoding="utf-8") as fin:
                lines = fin.readlines()
                for line in lines:

                    cleaned_line = line.split("$LABEL$")[0] if "$LABEL$" in line else line
                    cleaned_line = cleaned_line.replace("[B-ASP]", "").replace("[E-ASP]", "").strip()
                    if cleaned_line:
                        all_lines.append(cleaned_line)
        except FileNotFoundError:
            print(f"Warning: Dataset file not found: {fpath}")
        except Exception as e:
            print(f"Error loading {fpath}: {e}")


    seen = set()
    unique_ordered_lines = []
    for line in all_lines:
        if line not in seen:
            unique_ordered_lines.append(line)
            seen.add(line)
    return unique_ordered_lines

