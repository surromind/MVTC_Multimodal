def load_txt(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()
