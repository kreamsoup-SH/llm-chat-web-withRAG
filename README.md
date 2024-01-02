# LLM chatbot. web. with RAG
### *~~This Repository is currently under work.</br>The source code is not completed.~~*   
### *This repository is temporarily suspended..*
![demo/screenshot.png](https://github.com/kreamsoup-SH/llm-chat-web-withRAG/blob/main/demo/screenshot.png)

# How to use
## Install required packages
```
pip install torch
pip install transformers bitsandbytes accelerate
pip install streamlit
```

## Run
```
# streamlit run {your .py file}
streamlit run webui.py
```

~~or you can run webui.bat (on windows)~~
```
runtime\python.exe -m streamlit run webui.py
```

# To-do
- [ ] Redesign To-do list
- [ ] complete webui-classic.py : 70%
- [ ] complete webui-RAG.py : 50%
- [ ] "model load" button to apply new configs
- [ ] Fix the way to handle external data folder

- [ ] create batch file for windows user
- [ ] apply whisper to use STT
- [ ] apply ViTS to use TTS
- [ ] QLoRA check
- [ ] GPT like text stream
- [ ] 2bit quantization - [QuIP](https://github.com/Cornell-RelaxML/quip-sharp)

# Using...
transformers  
streamlit  
llama_index  

# References
https://github.com/nicknochnack/Llama2RAG  
https://docs.streamlit.io/  
https://docs.llamaindex.ai/en/stable/  
