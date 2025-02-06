# EzRAG
A lightweight script for running Retrieval Augmented Generation using DeepSeek. If you'd like to see what's going on under the hood, you can check out [my blog post](https://danny.bio/posts/20250125-retrieval-augmented-generation/).

## Requirements
1. Python 3.12.X, Python 3.13 currently has versioning conflicts
2. DeepSeek API key

## Getting started
1. Clone this repository: ```git clone https://github.com/danielgeiszler/EzRAG.git```
2. Install the requirements: ```uv pip install -r pyproject.toml```
3. Create a ```.env``` file containing your DeepSeek or OpenAI API key: ```API_KEY=your_key_here```
4. Run the script: ```python EzRag.py```
5. Open the given url in your browser
6. If the model produces incorrect information about where the Changzhou dialect is spoken, that means it is reading local data and it worked!

## Adding your own data
Currenly, only .txt, .docx, and .pdf files are supported. Adding your files to the ```data``` directory is sufficient for them to be included in the system.

## Coming soon
* Support for more models
* Python 3.13 support
