# BizGenie AI Engineer Technical Test – README

### Introduction
Welcome to the BizGenie AI Engineer Technical Test.
The goal is to evaluate your ability to build core AI features that BizGenie will use: similarity search, ranking, and AI microservice integration.

You do not need to build a large project. Focus on writing **clean, minimal, and working code** that demonstrates your technical skills.

---

### Provided Datasets
Three CSV files are included:

1. **freelancers.csv**
   - Fields: freelancer_id, name, skills, experience_years, rating
   - Sample data of freelancers and their profiles.

2. **projects.csv**
   - Fields: project_id, title, description
   - Sample client projects to be matched with freelancers.

3. **proposals.csv**
   - Fields: proposal_id, freelancer_id, relevance_score, rating, bid_price, success_rate
   - Sample freelancer proposals for ranking.

---

### Tasks

#### Part 1 – Similarity Search (Freelancer Matching)
- Generate embeddings for freelancer profiles using **OpenAI** or **HuggingFace models**.
- Store embeddings in **FAISS** (or Pinecone if you prefer).
- For a given project description, return the **top 3 most relevant freelancers**.

**Expected Output:**
A script/notebook that prints top 3 matches with similarity scores.

---

#### Part 2 – Ranking Function (Proposal Evaluation)
- Assume all proposals belong to the same project.
- Create a scoring function that ranks proposals in `proposals.csv`.
- Use relevance score, rating, bid price, and success rate.
- Normalize values and combine them into a single score.
- Output proposals ordered best → worst.

**Expected Output:**
A Python function with ranked proposals.

---

#### Part 3 – API Microservice (Integration)
- Wrap Part 1 or Part 2 into a FastAPI microservice.

**Endpoint:**
- POST /match_freelancers
- Input: { "project_description": "string" }
- Output:
```json
{
  "matches": [
    { "freelancer_id": "F001", "score": 0.87 },
    { "freelancer_id": "F004", "score": 0.82 },
    { "freelancer_id": "F006", "score": 0.79 }
  ]
}
```

---

### Deliverables
- Python code (scripts or notebooks).
- FastAPI service (with requirements.txt).
- A short README including:
  - Setup instructions.
  - Example request/response for the API.


**Timeline:**
You have **72 hours** from receiving this test to submit.

---

### Evaluation Criteria
- Correctness: Does the solution work?
- Code Quality: Is it clean, modular, and documented?
- Practicality: Is the approach realistic for production use?
- API Design: Is the endpoint clear and functional?

---

**Tip:** Keep things simple. Use small models and local FAISS unless you want to showcase Pinecone or other services.
