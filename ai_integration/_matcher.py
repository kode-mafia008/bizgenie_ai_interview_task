import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from dataclasses import dataclass
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