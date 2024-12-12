import requests
import base64
from PIL import Image
import io

# Server IP and endpoint
endpoint = f"https://nemoooooooooo--supir-upscaler-fastapi-app.modal.run/upscale_image"

# Utility to convert an image to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_string

# Path to the test image
test_image_path = "123.jpg"  # Replace with the path to your test image

# Convert the test image to base64
base64_image = image_to_base64(test_image_path)

# Request payload
payload = {
    "input": {"target_image": base64_image},
    "upscaling_factor": 2  # You can test with 3 as well
}

# Send the POST request
try:
    response = requests.post(endpoint, json=payload)
    print("request sent")
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Parse and process the response
    response_data = response.json()
    print("Response received from server:")
    print(response_data)

    # Save the returned base64 image to a file (optional)
    base64_output_image = response_data["output"]["image"].split(",")[1]  # Extract base64 part
    output_image_data = base64.b64decode(base64_output_image)
    with open("output_image.png", "wb") as output_file:
        output_file.write(output_image_data)
    print("Output image saved as 'output_image.png'")

except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")