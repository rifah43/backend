import os

TEMP_DIR = os.path.join(os.getcwd(), 'temp')  # Or any other path you want

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)  # Create the directory if it doesn't exist
