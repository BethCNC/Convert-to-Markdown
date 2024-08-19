import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import os
import re

def fetch_html(url):
    """Fetch the HTML content of the URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None

def extract_main_content(html):
    """Extract the main content from the HTML, focusing on tutorial-like structures."""
    soup = BeautifulSoup(html, 'html.parser')

    # Attempt to identify the main content area, focusing on common tags used in tutorials
    main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')

    if not main_content:
        # Fallback to the body if the above doesn't work
        main_content = soup.find('body')

    # Remove unnecessary elements like scripts, styles, footers, and navigation
    for element in main_content.find_all(['script', 'style', 'footer', 'nav', 'aside']):
        element.decompose()

    # Return the cleaned content
    return main_content

def extract_title(soup):
    """Extract the title or H1 from the HTML."""
    title_tag = soup.find('title')
    h1_tag = soup.find('h1')

    if h1_tag:
        return h1_tag.get_text().strip()
    elif title_tag:
        return title_tag.get_text().strip()
    else:
        return "output"

def format_to_markdown(content):
    """Convert the HTML content to markdown with a focus on preserving the original text."""
    # Convert HTML to markdown, ensuring that the conversion retains the exact wording and structure
    markdown_content = md(str(content), heading_style="ATX")

    # Further refine the markdown for common issues
    markdown_content = re.sub(r'\n{2,}', '\n\n', markdown_content)  # Ensure consistent spacing

    return markdown_content

def save_markdown(markdown_content, output_file):
    """Save the markdown content to a file."""
    with open(output_file, 'w') as file:
        file.write(markdown_content)
    print(f"Markdown content saved to {output_file}")

def sanitize_filename(filename):
    """Sanitize the filename to remove invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def url_to_markdown(url):
    """Complete process: URL to markdown file."""
    html = fetch_html(url)
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        main_content = extract_main_content(html)
        if main_content:
            markdown_content = format_to_markdown(main_content)
            title = extract_title(soup)
            sanitized_title = sanitize_filename(title)
            output_file = os.path.join(os.getcwd(), f"{sanitized_title}.md")
            save_markdown(markdown_content, output_file)
        else:
            print("Main content could not be extracted.")
    else:
        print("Failed to fetch HTML content.")

def main():
    choice = input("Enter 'url' to provide a URL or 'pdf' to provide a PDF file path: ").strip().lower()

    if choice == 'url':
        url = input("Please enter the URL: ").strip()
        url_to_markdown(url)
    elif choice == 'pdf':
        pdf_path = input("Please enter the PDF file path: ").strip()
        print("PDF handling not implemented in this enhanced version.")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
