from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

model.save_pretrained("./src/community/other_models/gte")