from langchain_community.document_loaders.pdf import PyPDFLoader

# test if langchain will work
loader = PyPDFLoader("Instrukcja.pdf")
pages = loader.load_and_split()
print(pages[0])
print(len(pages))