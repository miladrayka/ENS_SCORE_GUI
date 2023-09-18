
import os
import shutil

def delete_files_and_subdirectories(directory_path):
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
                else:
                    shutil.rmtree(entry.path)
            print("All files and subdirectories deleted successfully.")
    except OSError:
        print("Error occurred while deleting files and subdirectories.")
