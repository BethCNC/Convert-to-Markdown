# Convert URL or PDF to Markdown

## Overview

This project provides a tool to convert URLs into Markdown files. The primary purpose is to use these Markdown files for training language models, making it easier to work with structured and easily accessible content. The best use case for this tool is converting online tutorials and articles into Markdown format, which can then be used as training data for custom language models.

## Features

- **Convert URLs to Markdown:** Extracts the main content of a webpage and converts it into a well-formatted Markdown file.
- **PDF Support:** (Note: PDF handling is not yet implemented in this version.)

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/BethCNC/Convert-to-Markdown.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd Convert-to-Markdown
   ```

3. **Set Up the Virtual Environment:**

   Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install Required Packages:**

   Install the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Script:**

   Use the following command to run the script and convert a URL or PDF to Markdown:

   ```bash
   python3 src/convert_to_markdown.py
   ```

   - You will be prompted to provide a URL or PDF file path.
   - If you select a URL, the tool will extract the content and save it as a Markdown file.

## Example Use Case

The tool is particularly useful for converting online tutorials and articles into Markdown format. For example, you might want to convert a multi-part tutorial on Python programming into Markdown files that can be used to train a language model to provide accurate and contextual responses related to the tutorial's content.

## Future Enhancements

- **PDF Handling:** The ability to convert PDF documents into Markdown is planned for a future release.
- **Content Filtering:** More advanced content filtering and formatting options for more customizable Markdown output.

## Contributing

Feel free to fork this repository and submit pull requests for any enhancements or fixes.

## License

This project is licensed under the MIT License.
