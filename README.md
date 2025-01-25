# EzRAG
A lightweight script for running Retrieval Augmented Generation using DeepSeek

## Requirements
1. Python 3.12.X, Python 3.13 currently has versioning conflicts
2. DeepSeek API key

## Getting started
1. Clone this repository: ```git clone https://github.com/danielgeiszler/EzRAG.git```
2. Install the requirements: ```uv pip install -r requirements.txt```
3. Create a .env file containing your DeepSeek API key: ```DEEPSEEK_API_KEY=your_key_here```
4. Run the script: ```python EzRag.py```
5. Open the given url in your browser
6. If the model produces incorrect information about Chinese dialects, it worked!

## Adding your own data
Currenly, only .txt files are supported. Adding your .txt files to the ```data``` directory is sufficient for them to be included in the system.

## Known issues
* Some of the LangChain document loading libraries are known to hang in certain environments. So far, I have encountered this issue in GitBash and Jupyter Notebooks. Running in Powershell works fine.

## Coming soon
* Support for more document types
* Support for more models
* Bug fixes for other environments
* Python 3.13 support
