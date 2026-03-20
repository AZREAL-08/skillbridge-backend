# Sandbox: Run Skill-Sim in isolation
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('alvperez/skill-sim-model')
test_embedding = model.encode("React", convert_to_tensor=True)
print(test_embedding.shape)