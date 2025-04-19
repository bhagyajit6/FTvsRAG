**Abstract**

Large Language Models (LLMs) are trained on huge datasets, which allow them to answer questions from various domains. However, their expertise is confined to the data that they were trained on. 
In order to specialize LLMs in niche domains like healthcare, finance, law, and education, various training methods can be employed. 
Two of these commonly known approaches are Retrieval-Augmented Generation and model fine-tuning. Five models—Llama-3.1-8B, Gemma-2-9B, Mistral-7B-Instruct, Qwen2.5-7B, and Phi-3.5-Mini-Instruct—were fine-tuned on healthcare data. 
These models were trained using three distinct approaches: Retrieval-Augmented Generation alone, fine-tuning alone, and a combination of both. 
Our findings revealed that RAG and FT+RAG consistently outperformed FT alone across most models, particularly LLAMA and PHI. 
LLAMA and PHI excelled across multiple metrics, with Llama showing superior overall performance and Phi demonstrating strong RAG/FT+RAG capabilities. QWEN lagged behind in most metrics, while GEMMA and MISTRAL showed mixed results.

Keywords: Large Language Models, Healthcare, Retrieval-Augmented Generation, Fine-tuning

**Notebooks**
+ [Notebooks for LLM Fine tuning](Training)-contains 5 notebooks for each model and the code for their fine tuning.

+ [Responses from Gemma](Gemma_Responses)-contains three files with responses from FT, RAG, and FT+RAG versions of Gemma.
+ [Responses from Llama](Llama_Responses)-contains three files with responses from FT, RAG, and FT+RAG versions of Llama.
+ [Responses from Mistral](Mistral_Responses)-contains three files with responses from FT, RAG, and FT+RAG versions of Mistral.
+ [Responses from Phi](Phi_Responses)-contains three files with responses from FT, RAG, and FT+RAG versions of Phi.
+ [Responses from Qwen](Qwen_Responses)-contains three files with responses from FT, RAG, and FT+RAG versions of Qwen.

+ [Graphs](Graphs.ipynb)-contains the code to generate graphs.
+ [Retrieval-Augmented Generation](RAG.ipynb)-contains the code to implement RAG. 
+ [Qualitative Analysis](Qualitative_Analysis.ipynb)-contains the code to evaluate the LLMs.


**Models**
The Fine tuned 16-bit models are available in huggingface
[Gemma-2-9b-medquad](https://huggingface.co/bpingua/gemma-2-9b-medquad),
[Mistral-7b-instruct-v0.3-bnb-4bit-medquad](https://huggingface.co/bpingua/mistral-7b-instruct-v0.3-bnb-4bit-Medquad),
[Llama-3.1-8B-medquad](https://huggingface.co/bpingua/Llama-3.1-8B-medquad),
[Qwen2.5-7B-medquad](https://huggingface.co/bpingua/Qwen2.5-7B-Medquad-16bit),
[Phi-3.5-mini-instruct-medquad](https://huggingface.co/bpingua/Phi-3.5-mini-instruct-Medquad)

**Dataset**
2 types of Medquad dataset were used
[Original Medquad Dataset](https://huggingface.co/datasets/bpingua/medquad_cleaned)-cleaned version of original [MedQuAD](https://paperswithcode.com/dataset/medquad)
[Medquad Dataset ShareGPT](https://huggingface.co/datasets/bpingua/medquad_sharegpt_cleaned)-dataset in ShareGPT style.


