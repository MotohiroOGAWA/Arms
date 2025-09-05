import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

def plot_venn3(list_dict, title="Venn Diagram"):
    """
    Plot a Venn diagram from up to 3 named lists.
    Return a dict with region IDs and their counts.
    """
    sets = {name: set(lst) for name, lst in list_dict.items()}
    labels = list(sets.keys())
    values = list(sets.values())

    plt.figure(figsize=(6,6))

    if len(sets) == 2:
        v = venn2(values, set_labels=labels)
        region_ids = ["10", "01", "11"]
    elif len(sets) == 3:
        v = venn3(values, set_labels=labels)
        region_ids = ["100", "010", "001", "110", "101", "011", "111"]
    else:
        raise ValueError("Only 2 or 3 sets are supported for Venn diagrams.")

    plt.title(title)

    # collect counts
    counts = {}
    for rid in region_ids:
        label = v.get_label_by_id(rid)
        if label is not None:
            counts[rid] = int(label.get_text())
        else:
            counts[rid] = 0

    plt.show()
    return counts
