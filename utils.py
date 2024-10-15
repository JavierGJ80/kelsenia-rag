import os

def create_tuple_from_pdf_dir(dir:str) -> tuple[str]:
    file_paths: list[str] = []

    for file in os.listdir(dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(dir, file)
            file_paths.append(pdf_path)

    return tuple(file_paths)