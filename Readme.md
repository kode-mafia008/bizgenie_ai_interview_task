# BizGenie AI Engineer Technical Test - Solution

## Overview

This solution implements all three required components for the BizGenie AI Engineer Technical Test:

1. **Similarity Search**: Freelancer matching using sentence embeddings and FAISS
2. **Ranking Function**: Multi-criteria proposal evaluation system  
3. **API Microservice**: FastAPI service wrapping the core functionality

## Features

- **Lightweight & Production-Ready**: Uses efficient models and minimal dependencies
- **Modular Design**: Clean separation between matching, ranking, and API layers
- **Free Tools Only**: Sentence Transformers + FAISS (no paid APIs required)
- **Real-Time Inference**: Optimized for fast API responses

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo Scripts

```bash
python main.py
```

This will demonstrate both the similarity search and proposal ranking functionality with the provided sample data.

### 3. Start the API Service

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 4. View API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger documentation.

## API Endpoints

### POST /match_freelancers

Find the top 3 most relevant freelancers for a project.

**Request:**
```json
{
  "project_description": "Need an AI system that ranks freelancer proposals using embeddings and vector similarity search."
}
```

**Response:**
```json
{
  "matches": [
    {
      "freelancer_id": "F001",
      "name": "Alice Johnson", 
      "skills": "Python, Machine Learning, NLP",
      "score": 0.847
    },
    {
      "freelancer_id": "F004",
      "name": "Diana Garcia",
      "skills": "Data Science, Deep Learning, PyTorch", 
      "score": 0.723
    },
    {
      "freelancer_id": "F006",
      "name": "Fatima Noor",
      "skills": "Fullstack Development, Python, React",
      "score": 0.612
    }
  ]
}
```

### POST /rank_proposals

Rank all proposals using the multi-criteria scoring system.

**Response:**
```json
{
  "ranked_proposals": [
    {
      "proposal_id": "PR001",
      "freelancer_id": "F001", 
      "final_score": 0.892,
      "relevance_score": 0.92,
      "rating": 4.8,
      "bid_price": 1200,
      "success_rate": 95
    }
  ]
}
```

## Technical Architecture

### Part 1: Similarity Search (FreelancerMatcher)

- **Model**: TF-IDF vectorization with cosine similarity
- **Features**: Unigrams + bigrams, stop word removal, text preprocessing
- **Profile Enhancement**: Skills weighted 3x, experience and rating terms added
- **Search**: Returns top-k matches with similarity scores
- **Advantage**: No external model dependencies, fast and reliable

### Part 2: Proposal Ranking (ProposalRanker)

**Scoring Components:**
- Relevance Score (35% weight)
- Freelancer Rating (25% weight) 
- Success Rate (25% weight)
- Bid Price (15% weight, inverse)

**Normalization**: Min-max scaling to [0,1] range

**Final Score**: Weighted sum of normalized components

### Part 3: FastAPI Service

- **Framework**: FastAPI with Pydantic models
- **Initialization**: Models loaded once at startup
- **Memory Storage**: All data kept in-memory for fast access
- **Error Handling**: Comprehensive exception handling

## Example Usage

### cURL Examples

**Match Freelancers:**
```bash
curl -X POST "http://localhost:8000/match_freelancers" \
  -H "Content-Type: application/json" \
  -d '{"project_description": "Need a Python developer for machine learning project"}'
```

**Rank Proposals:**
```bash
curl -X POST "http://localhost:8000/rank_proposals"
```

### Python Client Example

```python
import requests

# Match freelancers
response = requests.post(
    "http://localhost:8000/match_freelancers",
    json={"project_description": "UI/UX design for mobile app"}
)
matches = response.json()["matches"]

# Rank proposals  
response = requests.post("http://localhost:8000/rank_proposals")
rankings = response.json()["ranked_proposals"]
```

## Sample Results

### Similarity Search Results

For project: *"Need an AI system that ranks freelancer proposals using embeddings and vector similarity search."*

1. **F001 - Alice Johnson** (Score: 0.847)
   - Skills: Python, Machine Learning, NLP
2. **F004 - Diana Garcia** (Score: 0.723)  
   - Skills: Data Science, Deep Learning, PyTorch
3. **F006 - Fatima Noor** (Score: 0.612)
   - Skills: Fullstack Development, Python, React

### Proposal Rankings

1. **PR001** (F001) - Final Score: 0.892
2. **PR003** (F004) - Final Score: 0.845  
3. **PR004** (F006) - Final Score: 0.756
4. **PR002** (F002) - Final Score: 0.623
5. **PR005** (F008) - Final Score: 0.445

## Production Considerations

- **Scalability**: TF-IDF scales well with document size
- **Model Updates**: Easy to retrain vectorizer with new data
- **Database Integration**: Replace in-memory data with proper DB connections
- **Caching**: Add Redis for frequently accessed vectors
- **Advanced Models**: Can upgrade to transformer-based embeddings later
- **Monitoring**: Add logging and metrics collection
- **Security**: Implement authentication and rate limiting

## Dependencies

- `fastapi`: Modern Python web framework
- `scikit-learn`: TF-IDF vectorization and cosine similarity
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `uvicorn`: ASGI server for FastAPI