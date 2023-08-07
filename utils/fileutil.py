import os


def get_files(dir_name) -> [str]:
    file_list = []
    for root, dirs, files in os.walk(dir_name):
        file_path = map(lambda file: os.path.join(root, file), files)
        file_list.extend(file_path)
    return file_list