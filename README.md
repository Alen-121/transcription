
# Youtube Transcript Summarizer

This project showcases a system designed to extract and analyze text YouTube video transcriptions. In this project a fine-tuned model will provide you a summary for a certain duration , which will provided by the user.


## Features

- Provide summary for a the given transcription
- Provide the content for user specified duration

### Note & issues:
- This project isn't complete because of lack data.
- I have firstly trained the mode using the transcription (.txt) and then fine-tuned using the json file which contents the the duration . The fine-tuned model needs lots of train .
- The accurasy may be low.
- I have firstly tried to train the model with certain words such as end , ending ,finish , close , closure , termination for understanding the context to the model. Due to lack of experience that didn't turned well.
## Frameworks and Libraries

Additional frameworks and libraries used in this project:


* ![Langchain](https://img.shields.io/badge/Langchain-blue?style=for-the-badge&logo=data:image/png;base64,iVBORwkgQAAQABTnk3Qn8+PwB/AAIAAAACXQFBQkAQA4KENsQAAAA1CAIACAAAAD8CAIAAACvCAIACAAAAgwCAIAAAC9CAIAAACzCAIAAAChCAIAAACnCAIAAACxCAIAAAC1CAIAAAC3CAIAAAC5CAIAAAC7CAIAAAC9CAIAAAC/CAIAAACHCAIAAACJCAIAAACLCAIAAACNCAIAAACPCAIAAACRCAIAAACTCAIAAACVCAIAAACXCAIAAACZCAIAAAbCAIAAAdCAIAAAgCAIAAAlCAIAAAnCAIAAArCAIAAAtCAIAAAufCAIAAAvCAIAAAxCAIAAAzCAIAAAB1CAIAAAB3CAIAAAB5CAIAAAB7CAIAAAB9CAIAAA/CAIAAAABhCAIAAAjCAIAAAjlCAIAAAjnCAIAAAjpCAIAAAjrCAIAAAtCAIAAAvCAIAAAxCAIAAAzCAIAAAB1CAIAAAB3CAIAAAB5CAIAAAB7CAIAAAB9CAIAAA/CAIAAAABhCAIAAAjCAIAAAjlCAIAAAjnCAIAAAjpCAIAAAjrCAIAAA)
## Installation

1.Clone the repository:

```bash
git clone https://github.com/Alen-121/transcription.git

```
2.Navigate to the project directory and install dependencies:

```bash
cd yt-transcription
pip install -r requirements.txt

```
3.Run the project
```bash

python transcript.py

```
    
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY` : https://aistudio.google.com/app/apikey

## 🚀 About Me
Hello! I'm Alen Sunny, . I'm passionate about learn ML, DL ,LLM and always eager to learn and grow in my field.

## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alen--sunny/)
