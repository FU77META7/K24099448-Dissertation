import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from pathlib import Path

# Custom dataset
class RelevanceDataset(Dataset):
    def __init__(self, json_path, model):
        self.model = model
        self.samples = []
        with open(json_path, "r") as f:
            data = json.load(f)
        for qa in data["question_answer_pairs"]:
            q = qa["question"]
            for s in qa["answer"].split("."):
                if s.strip():
                    score = np.random.uniform(0,1)  # <-- replace with real label if available
                    self.samples.append((q, s.strip(), score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, s, score = self.samples[idx]
        q_emb = self.model.encode(q, convert_to_tensor=True)
        s_emb = self.model.encode(s, convert_to_tensor=True)
        pair = torch.cat([q_emb, s_emb, torch.abs(q_emb - s_emb)], dim=-1)
        return pair, torch.tensor(score, dtype=torch.float)

# Regression model
class RelevanceRegressor(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3*emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output 0–1
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)

def train(json_path, epochs=3, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    sbert = SentenceTransformer(model_name)
    dataset = RelevanceDataset(json_path, sbert)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    regressor = RelevanceRegressor(emb_dim=sbert.get_sentence_embedding_dimension()).to(device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        regressor.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = regressor(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(loader):.4f}")

    torch.save(regressor.state_dict(), "relevance_regressor.pt")
    print("✅ Model saved as relevance_regressor.pt")

if __name__ == "__main__":
    train("reformatted_gpt_o1_responses_with_labels.json")