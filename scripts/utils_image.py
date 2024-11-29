import base64
from PIL import Image
from io import BytesIO

def Image2base64(image_path:str) -> str:
    """
    Convert an image file to a base64 encoded string.
    
    :param image_path: Path to the image file.
    :return: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(
            image_file.read()
        ).decode('utf-8')
    return encoded_string

def base64toImage(base64_string:str, output_path:str):
    """
    Convert a base64 encoded string to an image file.
    
    :param base64_string: Base64 encoded string of the image.
    :param output_path: Path to save the decoded image `.png` file.
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image.save(output_path)

# Example usage:
# base64_str = Image2base64("path_to_image.jpg")
# base64toImage(base64_str, "output_image.jpg")

if __name__ == "__main__":
    base64_str = Image2base64("ollama.png")
    print(base64_str)

# python utils_image.py