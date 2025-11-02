from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
text_e = model.encode('This is a test sentence')


