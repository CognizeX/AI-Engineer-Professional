from pathlib import Path
from pptxtopdf import convert

# Directories
input_dir = Path("public")
output_dir = input_dir / "pdf"
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Convert all PPTX files
    for pptx_file in input_dir.glob("*.pptx"):
        pdf_file = output_dir / f"{pptx_file.stem}.pdf"

        print(f"Converting {pptx_file} -> {pdf_file}")
        convert(str(pptx_file), str(pdf_file))

    print("Done!")