# rag-pipeline

A RAG Pipeline for retrival of information related to Food Ingredients provided query using the mixed-bread embedding model.
This project focuses on building a RAG setup to generate quality embeddings,  \
and to apply various techniques like multiprocessing and pyTorch DDP inorder  \
improve the time taken to generate these embeddings. 
The project focuses on:
- Scraping and parsing of large wikipedia articles, over 50K articles exclusively related to food ingredients.
- We employ python's multiprocessing module to speed up the parsing process and save the final parsed results as pickle dumps.
- Then we divide the articles in chuncks of size ~300 tokens on average based on context ( end of sentences or paragraphs )
- With the mixed bread (mxbai) embedding model, we embed these chunks to generate quality embeddings. 
- We used pytorch's DDP module to speed up the embedding process using 2 GPUs on NYU HPC.
- We save the embeddings, then index them with Faiss to use it as a vector database.
- Evaluating the quality of generated embeddings. 
- Finally, we integrate the retrival pipeline with gemma for fine tuning and inference. 

# To-Run the Code:



