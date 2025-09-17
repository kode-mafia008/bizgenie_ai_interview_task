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
 
## 🛠️ Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git

## 📦 Installation & Setup

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/kode-mafia008/bizgenie_ai_interview_task.git
   cd bizgenie_ai_interview_task
   ```

2. **Make the entry script executable**
   ```bash
   chmod +x entryPoint.sh
   ```

3. **Start the application**
   ```bash
   ./entryPoint.sh
   ```

   This will:
   - Create a `.env` file if it doesn't exist
   - Build the Docker image with all dependencies
   - Start the container in detached mode
   - Display application logs

4. **Access the application**
   - API Documentation: http://localhost:8080/docs
   - Main endpoint: http://localhost:8080/

### Option 2: Local Development

1. **Clone and navigate to project**
   ```bash
   git clone https://github.com/kode-mafia008/bizgenie_ai_interview_task.git
   cd bizgenie_ai_interview_task
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
PORT=8080
ENVIRONMENT=development
PYTHONPATH=/app
```

### Docker Configuration

The application uses the following Docker setup:

- **Base Image**: `python:3.12-slim-bullseye`
- **Port**: 8080
- **Working Directory**: `/app`
- **User**: Non-root user for security

## 🚀 Usage

### API Endpoints

Once the application is running, visit http://localhost:8080/docs for interactive API documentation.

#### Core Endpoints

- `GET /` - Health check endpoint
- `POST /recommend` - Get personalized service recommendations
- `POST /chatbot` - Interact with the AI chatbot

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change port in docker-compose.yaml or kill existing process
   lsof -ti:8080 | xargs kill -9
   ```

2. **Model loading errors**
   - The system includes fallback mode for missing models
   - Check logs for "fallback mode" messages
   - Ensure model files exist in `ai_integration/models/`

3. **Docker build failures**
   ```bash
   # Clean Docker cache
   docker system prune -a
   docker compose build --no-cache
   ```

4. **Dependency issues**
   ```bash
   # Rebuild with fresh dependencies
   pip install --upgrade -r requirements.txt
   ```



## 📈 Monitoring

### Application Health
- Check `/` endpoint for basic health status
- Monitor Docker logs: `docker compose logs -f`
- API documentation: `/docs` endpoint


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
curl -X POST "http://localhost:8080/match_freelancers" \
  -H "Content-Type: application/json" \
  -d '{"project_description": "Need a Python developer for machine learning project"}'
```

**Rank Proposals:**
```bash
curl -X POST "http://localhost:8080/rank_proposals"
```

### Python Client Example

```python
import requests

# Match freelancers
response = requests.post(
    "http://localhost:8080/match_freelancers",
    json={"project_description": "UI/UX design for mobile app"}
)
matches = response.json()["matches"]

# Rank proposals  
response = requests.post("http://localhost:8080/rank_proposals")
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request with detailed description


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Docker logs for error details
3. Ensure all dependencies are properly installed
4. Verify model files are present and accessible

## 🔄 Version History

- **v1.0.0**: Initial release with recommendation system and chatbot
- **v1.1.0**: Added Docker support and improved error handling
- **v1.2.0**: Enhanced model performance and fallback mechanisms