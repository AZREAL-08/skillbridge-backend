# Sandbox: Run Skill-Sim in isolation
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Load the skills dataset
df = pd.read_csv("../dataset/ESCO/skills_en.csv")

# Extract the list of valid skills (dropping missing ones)
skills = df['preferredLabel'].dropna().tolist()

print(f"Loaded {len(skills)} skills from the CSV.")

# Load the model
model = SentenceTransformer('alvperez/skill-sim-model')

# Encode the target query
query = "React"
query_embedding = model.encode(query, convert_to_tensor=True)

# Encode all skills from the CSV
print("Encoding all skills from the CSV ...")
skill_embeddings = model.encode(skills, convert_to_tensor=True, show_progress_bar=True)

# Calculate cosine similarities
cos_scores = util.cos_sim(query_embedding, skill_embeddings)[0]

# Find the skill with the highest similarity score
best_match_idx = torch.argmax(cos_scores).item()
best_skill = skills[best_match_idx]
best_score = cos_scores[best_match_idx].item()

print(f"\nQuery: '{query}'")
print(f"Most similar skill: '{best_skill}' (Score: {best_score:.4f})")
