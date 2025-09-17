# BizGenie AI Engineer Technical Test - Complete Solution

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import json
from fastapi import FastAPI
from pydantic import BaseModel
import re
import logging

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FreelancerMatch:
    freelancer_id: str
    name: str
    skills: str
    score: float

@dataclass
class RankedProposal:
    proposal_id: str
    freelancer_id: str
    final_score: float
    relevance_score: float
    rating: float
    bid_price: int
    success_rate: int

class ProjectRequest(BaseModel):
    project_description: str

class MatchResponse(BaseModel):
    matches: List[Dict[str, Any]]

# =============================================================================
# Part 1: Similarity Search (Freelancer Matching)
# =============================================================================

class FreelancerMatcher:
    """
    Handles freelancer matching using TF-IDF vectorization and cosine similarity.
    This approach avoids dependency issues while still providing effective matching.
    """
    
    def __init__(self, max_features: int = 1000):
        """Initialize with TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
        self.freelancer_vectors = None
        self.freelancers_df = None
        self.freelancer_profiles = None
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching."""
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def load_freelancers(self, csv_path: str = None, df: pd.DataFrame = None):
        """Load freelancer data from CSV or DataFrame."""
        if df is not None:
            self.freelancers_df = df
        else:
            self.freelancers_df = pd.read_csv(csv_path)
        
        # Create comprehensive profiles for vectorization
        self.freelancer_profiles = []
        for _, row in self.freelancers_df.iterrows():
            # Emphasize skills and experience in the profile
            skills_repeated = f"{row['skills']} " * 3  # Give more weight to skills
            experience_terms = f"experienced developer {row['experience_years']} years professional"
            rating_terms = f"highly rated excellent quality {row['rating']}" if row['rating'] >= 4.5 else f"rated {row['rating']}"
            
            profile = f"{row['name']} {skills_repeated} {experience_terms} {rating_terms}"
            profile = self.preprocess_text(profile)
            self.freelancer_profiles.append(profile)
    
    def build_index(self):
        """Generate TF-IDF vectors for freelancer profiles."""
        logging.info("Generating TF-IDF vectors for freelancer profiles...")
        self.freelancer_vectors = self.vectorizer.fit_transform(self.freelancer_profiles)
        logging.info(f"Built TF-IDF index with {len(self.freelancer_profiles)} freelancers")
    
    def find_matches(self, project_description: str, top_k: int = 3) -> List[FreelancerMatch]:
        """Find top-k most similar freelancers for a project."""
        if self.freelancer_vectors is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Preprocess and vectorize project description
        processed_description = self.preprocess_text(project_description)
        query_vector = self.vectorizer.transform([processed_description])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.freelancer_vectors)[0]
        
        # Get top-k matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Format results
        matches = []
        for idx in top_indices:
            freelancer_row = self.freelancers_df.iloc[idx]
            match = FreelancerMatch(
                freelancer_id=freelancer_row['freelancer_id'],
                name=freelancer_row['name'],
                skills=freelancer_row['skills'],
                score=float(similarities[idx])
            )
            matches.append(match)
        
        return matches

# =============================================================================
# Part 2: Ranking Function (Proposal Evaluation)
# =============================================================================

class ProposalRanker:
    """
    Ranks freelancer proposals using multiple criteria with normalization.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize with scoring weights."""
        self.weights = weights or {
            'relevance_score': 0.35,
            'rating': 0.25,
            'success_rate': 0.25,
            'bid_price': 0.15  # Lower price is better
        }
    
    def normalize_column(self, values: np.array, reverse: bool = False) -> np.array:
        """Normalize values to 0-1 range. If reverse=True, lower values get higher scores."""
        min_val, max_val = values.min(), values.max()
        
        if max_val == min_val:
            return np.ones_like(values)
        
        normalized = (values - min_val) / (max_val - min_val)
        
        if reverse:
            normalized = 1 - normalized
        
        return normalized
    
    def rank_proposals(self, csv_path: str = None, df: pd.DataFrame = None) -> List[RankedProposal]:
        """Rank proposals using weighted scoring."""
        if df is not None:
            proposals_df = df
        else:
            proposals_df = pd.read_csv(csv_path)
        
        # Normalize each scoring component
        norm_relevance = self.normalize_column(proposals_df['relevance_score'].values)
        norm_rating = self.normalize_column(proposals_df['rating'].values)
        norm_success = self.normalize_column(proposals_df['success_rate'].values)
        norm_price = self.normalize_column(proposals_df['bid_price'].values, reverse=True)  # Lower price is better
        
        # Calculate weighted final scores
        final_scores = (
            norm_relevance * self.weights['relevance_score'] +
            norm_rating * self.weights['rating'] +
            norm_success * self.weights['success_rate'] +
            norm_price * self.weights['bid_price']
        )
        
        # Create ranked results
        proposals_df = proposals_df.copy()
        proposals_df['final_score'] = final_scores
        proposals_df = proposals_df.sort_values('final_score', ascending=False)
        
        # Format results
        ranked_proposals = []
        for _, row in proposals_df.iterrows():
            proposal = RankedProposal(
                proposal_id=row['proposal_id'],
                freelancer_id=row['freelancer_id'],
                final_score=float(row['final_score']),
                relevance_score=float(row['relevance_score']),
                rating=float(row['rating']),
                bid_price=int(row['bid_price']),
                success_rate=int(row['success_rate'])
            )
            ranked_proposals.append(proposal)
        
        return ranked_proposals

# =============================================================================
# Part 3: FastAPI Microservice
# =============================================================================

# Initialize components
app = FastAPI(title="BizGenie AI Matching Service", version="1.0.0")
matcher = FreelancerMatcher()
ranker = ProposalRanker()

# Sample data (in production, this would come from a database)
FREELANCERS_DATA = pd.DataFrame([
    {"freelancer_id": "F001", "name": "Alice Johnson", "skills": "Python, Machine Learning, NLP", "experience_years": 5, "rating": 4.8},
    {"freelancer_id": "F002", "name": "Bob Smith", "skills": "Laravel, PHP, MySQL", "experience_years": 4, "rating": 4.5},
    {"freelancer_id": "F003", "name": "Charlie Lee", "skills": "React, Node.js, Frontend Development", "experience_years": 3, "rating": 4.2},
    {"freelancer_id": "F004", "name": "Diana Garcia", "skills": "Data Science, Deep Learning, PyTorch", "experience_years": 6, "rating": 4.9},
    {"freelancer_id": "F005", "name": "Ethan Patel", "skills": "UI/UX Design, Figma, Adobe XD", "experience_years": 2, "rating": 4.1},
    {"freelancer_id": "F006", "name": "Fatima Noor", "skills": "Fullstack Development, Python, React", "experience_years": 4, "rating": 4.6},
    {"freelancer_id": "F007", "name": "George Brown", "skills": "DevOps, Docker, Kubernetes", "experience_years": 7, "rating": 4.7},
    {"freelancer_id": "F008", "name": "Hina Khan", "skills": "Data Analysis, SQL, Power BI", "experience_years": 3, "rating": 4.3}
])

PROPOSALS_DATA = pd.DataFrame([
    {"proposal_id": "PR001", "freelancer_id": "F001", "relevance_score": 0.92, "rating": 4.8, "bid_price": 1200, "success_rate": 95},
    {"proposal_id": "PR002", "freelancer_id": "F002", "relevance_score": 0.75, "rating": 4.5, "bid_price": 1000, "success_rate": 90},
    {"proposal_id": "PR003", "freelancer_id": "F004", "relevance_score": 0.88, "rating": 4.9, "bid_price": 1500, "success_rate": 97},
    {"proposal_id": "PR004", "freelancer_id": "F006", "relevance_score": 0.85, "rating": 4.6, "bid_price": 1100, "success_rate": 92},
    {"proposal_id": "PR005", "freelancer_id": "F008", "relevance_score": 0.7, "rating": 4.3, "bid_price": 900, "success_rate": 85}
])

# Initialize matcher with data
matcher.load_freelancers(df=FREELANCERS_DATA)
matcher.build_index()

@app.on_event("startup")
async def startup_event():
    """Initialize the matching system on startup."""
    logging.info("BizGenie AI Matching Service started successfully!")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "BizGenie AI Matching Service is running!"}

@app.post("/match_freelancers", response_model=MatchResponse)
async def match_freelancers(request: ProjectRequest):
    """
    Find the top 3 most relevant freelancers for a given project description.
    
    Args:
        request: ProjectRequest containing project_description
        
    Returns:
        MatchResponse with top 3 freelancer matches and similarity scores
    """
    try:
        matches = matcher.find_matches(request.project_description, top_k=3)
        
        response_matches = [
            {
                "freelancer_id": match.freelancer_id,
                "name": match.name,
                "skills": match.skills,
                "score": round(match.score, 3)
            }
            for match in matches
        ]
        
        return MatchResponse(matches=response_matches)
    
    except Exception as e:
        return {"error": f"Failed to match freelancers: {str(e)}"}

@app.post("/rank_proposals")
async def rank_proposals():
    """
    Rank all proposals using the weighted scoring system.
    
    Returns:
        List of ranked proposals with scores
    """
    try:
        ranked = ranker.rank_proposals(df=PROPOSALS_DATA)
        
        response = [
            {
                "proposal_id": proposal.proposal_id,
                "freelancer_id": proposal.freelancer_id,
                "final_score": round(proposal.final_score, 3),
                "relevance_score": proposal.relevance_score,
                "rating": proposal.rating,
                "bid_price": proposal.bid_price,
                "success_rate": proposal.success_rate
            }
            for proposal in ranked
        ]
        
        return {"ranked_proposals": response}
    
    except Exception as e:
        return {"error": f"Failed to rank proposals: {str(e)}"}

# =============================================================================
# Demo and Testing Functions
# =============================================================================

def demo_similarity_search():
    """Demo the freelancer matching functionality."""
    logging.info("=== PART 1: SIMILARITY SEARCH DEMO ===")
    
    # Initialize matcher
    matcher_demo = FreelancerMatcher()
    matcher_demo.load_freelancers(df=FREELANCERS_DATA)
    matcher_demo.build_index()
    
    # Test projects
    test_projects = [
        "Need an AI system that ranks freelancer proposals using embeddings and vector similarity search.",
        "Looking for a UI/UX expert to improve dashboard usability with Figma and Adobe XD.",
        "Require a Laravel developer with MySQL knowledge to build secure APIs."
    ]
    
    for i, project_desc in enumerate(test_projects, 1):
        logging.info(f"\nProject {i}: {project_desc}")
        logging.info("Top 3 matches:")
        
        matches = matcher_demo.find_matches(project_desc, top_k=3)
        for j, match in enumerate(matches, 1):
            logging.info(f"  {j}. {match.freelancer_id} - {match.name}")
            logging.info(f"     Skills: {match.skills}")
            logging.info(f"     Similarity Score: {match.score:.3f}")
            logging.info()

def demo_proposal_ranking():
    """Demo the proposal ranking functionality."""
    logging.info("=== PART 2: PROPOSAL RANKING DEMO ===")
    
    ranker_demo = ProposalRanker()
    ranked_proposals = ranker_demo.rank_proposals(df=PROPOSALS_DATA)
    
    logging.info("Ranked Proposals (Best to Worst):")
    logging.info("-" * 80)
    
    for i, proposal in enumerate(ranked_proposals, 1):
        logging.info(f"{i}. {proposal.proposal_id} (Freelancer: {proposal.freelancer_id})")
        logging.info(f"   Final Score: {proposal.final_score:.3f}")
        logging.info(f"   Relevance: {proposal.relevance_score}, Rating: {proposal.rating}")
        logging.info(f"   Bid: ${proposal.bid_price}, Success Rate: {proposal.success_rate}%")
        logging.info()

if __name__ == "__main__":
    # Run demos
    demo_similarity_search()
    demo_proposal_ranking()
    
    logging.info("=== API SERVICE ===")
    logging.info("To start the FastAPI service, run:")
    logging.info("uvicorn main:app --reload --port 8000")
    logging.info("\nExample API request:")
    logging.info("POST http://localhost:8000/match_freelancers")
    logging.info('{"project_description": "Need a Python developer for machine learning project"}')