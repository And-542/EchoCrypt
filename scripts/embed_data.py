import os

def text_to_binary(text, space_separated=True):
    """
    Converts a given text string into its binary representation.

    Args:
        text (str): The input text string.
        space_separated (bool): If True, separates each byte with a space.

    Returns:
        str: The binary representation of the text.
    """
    if space_separated:
        return ' '.join(format(ord(char), '08b') for char in text)
    else:
        return ''.join(format(ord(char), '08b') for char in text)

def save_binary_to_file(text, folder_path, filename="binary_output.txt", space_separated=True):
    """
    Converts text to binary and saves it to a file in the specified folder.

    Args:
        text (str): The input text string.
        folder_path (str): Folder to save output. Will be created if missing.
        filename (str): Output file name.
        space_separated (bool): If True, space between each byte in binary.

    Returns:
        str: The binary string saved to the file.
    """
    binary_data = text_to_binary(text, space_separated)

    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 Directory ensured: {folder_path}")
    except OSError as e:
        print(f"❌ Error creating directory {folder_path}: {e}")
        return None

    file_path = os.path.join(folder_path, filename)

    try:
        with open(file_path, 'w') as f:
            f.write(binary_data)
        print(f"✅ Binary saved to: {file_path}")
        return binary_data
    except IOError as e:
        print(f"❌ Error writing to file {file_path}: {e}")
        return None

# ------------------ Run this in VS Code / PyCharm ------------------

if __name__ == "__main__":
    # 🔐 Message to embed
    input_text = "Hello Andrew"

    # 📁 Output directory and file
    output_folder = r"D:\EchoCrypt\EchoCrypt\data\binary"
    output_filename = "echocrypt_message.txt"

    # 📝 Convert and save
    binary = save_binary_to_file(
        text=input_text,
        folder_path=output_folder,
        filename=output_filename,
        space_separated=True  # Set to False if you want no spaces
    )

    # 🔍 Show binary
    if binary:
        print(f"\n📤 Binary content:\n{binary}")
