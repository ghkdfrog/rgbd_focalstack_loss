import json
import sys

with open('gradient_matching_focal-Copy1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('extracted_nb.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            out.write(f"# ----- CODE CELL {i} -----\n")
            out.write("".join(cell['source']) + "\n\n")
        elif cell['cell_type'] == 'markdown':
            out.write(f"# ----- MARKDOWN CELL {i} -----\n")
            for line in cell['source']:
                out.write("# " + line.strip() + "\n")
            out.write("\n")
