# vision_server
Vision server for Ditto [assistant](http://github.com/ditto-assistant/assistant) clients. Holds models for image captioning and image Q/A. This server is used by the Image LLM Agent in [nlp_server](http://github.com/ditto-assistant/nlp_server).

# Installation and Running Locally
1. navigate into the project directory and run `pip install -r requirements.txt`
2. run `python main.py` to start the vision server. The server will be running on port 22032.

## Image Captioning and Q/A Models
The image captioning and Q/A models are used when [assistant](http://github.com/ditto-assistant/assistant) clients are prompted to describe an image. The image captioning model is used to generate a caption for the image, and the Q/A model is used to answer questions about the image. The models used are hosted on HuggingFace's model hub. The models are downloaded and cached locally when the vision server is started.

# LLM Agent over an Image (Demonstration)
Below is a demonstration of how [assistant](http://github.com/ditto-assistant/assistant) can use the [nlp_server](http://github.com/ditto-assistant/nlp_server)'s Image (Retrieval Augmented Generation) RAG Agent to carry out a conversation over an image. The RAG Agent is a few-shot prompting model that gives the agent access to the captioning and image Q/A models to generate responses. The RAG Agent is used when the user asks the assistant something requiring context from the assistan's environment.

## Example Image
![Example Image](./modules/river-sunset.png)


## Conversation
Below shows the LLM Agent's thought process when answering a user's query about the image. The LLM Agent first generates a caption for the image, then uses the Q/A model to answer the user's questions about the image. The LLM Agent then uses the caption and Q/A model however it sees fit to generate a response to the user's query. Anything with <> surrounding it is a decision made by the LLM Agent.
```
User's Query: Can you describe this image? I want to know where it is, what time of day it is, and what the weather is like.

Caption Model: landscape with a river and a tree vector art illustration

LLM Agent's Decisions:
<QA> Where is the image taken?
<QA Response> outside
<QA> What is the weather like?
<QA Response> sunny
<QA> What time of day is it?
<QA Response> evening
<DONE> I see a landscape with a river and a tree. The image was taken outside during the evening. The weather is sunny.

Final Response: I see a landscape with a river and a tree. The image was taken outside during the evening. The weather is sunny.
```
