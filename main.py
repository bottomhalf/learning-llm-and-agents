from src.service.simple_langchain import SimpleLangchain

simpleLangChain = SimpleLangchain()
simpleLangChain.document_loader()

simpleLangChain.split_document()

query = "what Neuroscience says about dream"
simpleLangChain.get_answer(query)
