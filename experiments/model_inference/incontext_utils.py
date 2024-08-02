from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

def concatenate_images_with_text(data: List[Dict[str, str]], output_path) -> str:
    """
    Concatenates images horizontally with space in between and adds two lines of text below each image,
    based on an array of dictionaries where each dictionary contains an image URL, a question, and an answer.

    Parameters:
    data (List[Dict[str, str]]): List of dictionaries, each containing 'image_url', 'question', and 'answer' keys.

    Returns:
    str: Path to the concatenated image file.
    """
    
    # Load images and texts
    images = []
    questions = []
    answers = []
    for item in data:
        url = item.get("image_url")
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        images.append(img)
        questions.append(item.get("question"))
        answers.append(item.get("answer"))

    # Define the spacing between images and the size of the text area
    spacing = 20
    text_area_height = 120

    # Calculate total width and height of the final image
    total_width = sum(image.width for image in images) + spacing * (len(images) - 1)
    total_height = max(image.height for image in images) + text_area_height

    # Create a new image with the calculated dimensions
    concatenated_img = Image.new("RGB", (total_width, total_height), "white")

    # Load a font
    font_size = 20
    # font = ImageFont.load_default(font_size)
    font = ImageFont.truetype("fonts/arial.ttf", font_size)
    bold_font = ImageFont.truetype("fonts/arialbd.ttf", font_size)
    title_font = ImageFont.truetype("fonts/arialbd.ttf", 40)

    # Initialize ImageDraw to add text
    draw = ImageDraw.Draw(concatenated_img)

    # Current width position to paste the next image
    current_width = 0
    image_y_offset = 35

    for i, image in enumerate(images):
        
        # Paste the image onto the concatenated image
        concatenated_img.paste(image, (current_width, font_size + image_y_offset))
        current_width += image.width + spacing

        # Add the text below the image
        question_text = questions[i]
        answer_text = answers[i]

        # Calculate text position
        text_x = current_width - image.width - spacing // 2
        text_y = image.height + 20 + image_y_offset

        # Draw the text
        draw.text((text_x + spacing // 2, text_y), "Q: ", fill="black", font=bold_font)
        draw.text((text_x + spacing // 2 + 30, text_y), question_text, fill="black", font=font)

        draw.text((text_x + spacing // 2, text_y + font_size + 10), "A: ", fill="black", font=bold_font)
        draw.text((text_x + spacing // 2 + 30, text_y + font_size + 10), answer_text, fill="black", font=font)

    title_text = "Examples"
    draw.text(( (current_width // 2) - ((len(title_text)*12) // 2) , 5), title_text, fill="black", font=title_font)
    # Save the concatenated image
    concatenated_img.save(output_path)
    
    return output_path


if __name__ == "__main__":
    data = [
        {
            "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/multi_col_133.png',
            "question": 'How many people are forecast to be occasional viewers of eSports by 2024?',
            "answer": '291.6'
        },
        {
            "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/two_col_100022.png',
            "question": 'What was the second-most-shared news page on Facebook in January 2017?',
            "answer": 'CNN'
        },
        {
            "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/two_col_6132.png',
            "question": 'How many widowed people lived in Canada in 2000?',
            "answer": '1.55'
        },
        {
            "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/two_col_1365.png',
            "question": 'What percentage of people infected with the coronavirus were female?',
            "answer": '51.1'
        }
    ]

    output_path = concatenate_images_with_text(data, "./incontext_img/chartqa_incontext.png")
