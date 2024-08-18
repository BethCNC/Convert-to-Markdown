# Hello There!

Congrats on starting up a new project in VS Code using Python and GitHub for version control. You are almost ready to create your amazing new script. There are just a couple of things to do before you start writing your code.

## 1. Activate Your Virtual Environment

Before you begin coding, activate your virtual environment with this command:

source venv/bin/activate

This ensures that any packages you install are scoped to this project only.

## 2. Add and Install Dependencies

If your project requires specific Python packages, add them to `requirements.txt` and then install them using the following command:

pip install -r requirements.txt

**Tip:** Whenever you add new dependencies, update `requirements.txt` with the following command:

pip freeze > requirements.txt

This keeps your project dependencies consistent.

## 3. Create a Git Repository and Make Your First Push

If you haven't connected your project to a remote Git repository (e.g., GitHub), follow these steps:

**Make sure you replace 'yourrepositoryname' with the actual name of your GitHub repository.**

1. Create the repository on GitHub (if you haven't done so already).
2. Set the remote URL and push your code:

git remote add origin https://github.com/bethcnc/###ADDPROJECTNAME###.git
git push -u origin master

## 4. Add API Keys and Secret Codes to `.env`

For any sensitive information like API keys and secret codes, use the `.env` file. Make sure not to commit this file to version control:

**Here is the link to obtain an API Key for Open AI: [OpenAI API Keys](https://platform.openai.com/api-keys)**

# Example .env file
API_KEY=your_api_key
SECRET_KEY=your_secret_key

## 5. Start Coding Your Innovative Script!

Now you're ready to create an innovative script to make your life easier with Python. Happy coding!

## **Remember:**

- **Regularly Push to GitHub**: After making significant changes to your code, make sure to commit and push them to your GitHub repository.
- **Document Your Code**: Maintain good documentation to help you or others understand your code later.
- **Test Your Code**: Write tests to ensure your code works as expected, especially as it grows.

---

You’re all set! Best of luck with your project.
