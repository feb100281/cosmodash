from pathlib import Path

BASE_DIR = Path(__file__).parent

output_file = BASE_DIR / "icons-list.txt"

with output_file.open("w", encoding="utf-8") as f:
    for file in BASE_DIR.iterdir():
        if file.is_file():
            name = file.name
            if name in ['collect.py','icons-list.txt']:
                continue
            parts = name.split('--')
            if len(parts) > 1:
                name = parts[1]
            name = name.replace('.svg','').replace('-','_')
            
            f.write(f'{name} = "solar/{file.name}"' + "\n")