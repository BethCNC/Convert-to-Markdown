import requests
from bs4 import BeautifulSoup
import markdownify
import os

def url_to_markdown(url, output_folder):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Convert the HTML to Markdown
    markdown_text = markdownify.markdownify(str(soup), heading_style="ATX")

    # Determine the output file path
    output_file = os.path.join(output_folder, 'output.md')
    
    # Write the Markdown to a file
    with open(output_file, 'w') as file:
        file.write(markdown_text)

    print(f"Successfully converted {url} to {output_file}")

if __name__ == "__main__":
    url = input("Enter the URL you want to convert: ")
    output_folder = "Output_Files"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    url_to_markdown(url, output_folder)