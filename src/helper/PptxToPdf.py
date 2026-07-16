from pathlib import Path
import shutil
from pptxtopdf import convert

root_dir = Path(__file__).resolve().parent.parent.parent
input_dir = root_dir / "public"
output_dir = root_dir / "public" / "pdf"

output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for pptx_file in input_dir.rglob("*.pptx"):
        temp_pdf_folder = output_dir / pptx_file.stem
        expected_pdf = temp_pdf_folder / f"{pptx_file.stem}.pdf"
        final_pdf = output_dir / f"{pptx_file.stem}.pdf"

        print(f"Converting {pptx_file.name}...")
        
        convert(str(pptx_file), str(output_dir))
        
        if temp_pdf_folder.exists() and temp_pdf_folder.is_dir():
            if expected_pdf.exists():
                shutil.move(str(expected_pdf), str(final_pdf))
            
            shutil.rmtree(temp_pdf_folder)

    print("Done! All files are now in:", output_dir)