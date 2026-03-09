import json
import difflib

def extract_source(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    lines = []
    for cell in nb['cells']:
        for line in cell.get('source', []):
            lines.append(line if line.endswith('\n') else line + '\n')
        lines.append('\n')
    return lines

a_path = 'gradient_matching_focal.ipynb'
b_path = 'gradient_matching_focal_coc.ipynb'

a_src = extract_source(a_path)
b_src = extract_source(b_path)

diff = list(difflib.unified_diff(a_src, b_src, fromfile=a_path, tofile=b_path))

with open('nb_diff.txt', 'w', encoding='utf-8') as f:
    if not diff:
        f.write("No differences found.\n")
    else:
        f.writelines(diff)
print("Diff extraction complete.")
