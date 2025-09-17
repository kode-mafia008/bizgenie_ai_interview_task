from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from data_loader import FREELANCERS_DATA
import logging

router = APIRouter()

class ProjectRequest(BaseModel):
    project_description: str = "Require a Laravel developer with MySQL knowledge to build secure APIs."

class MatchResponse(BaseModel):
    matches: List[Dict[str, Any]]

@router.on_event("startup")
async def startup_event():
    """Initialize the matcher with data on startup."""
    from main import matcher
    logging.info("Loading freelancer profiles...")
    matcher.load_freelancers(df=FREELANCERS_DATA)
    logging.info("Loaded freelancer profiles")
    logging.info("Generating TF-IDF vectors for freelancer profiles...")
    matcher.build_index()
    logging.info("Built TF-IDF index with freelancers")
    logging.info("BizGenie AI Matching Service started successfully!")

@router.post("/freelancers", response_model=MatchResponse)
async def match_freelancers(request: ProjectRequest):
    """
    Find the top 3 most relevant freelancers for a given project description.
    
    Args:
        request: ProjectRequest containing project_description
        
    Returns:
        MatchResponse with top 3 freelancer matches and similarity scores
    """
    try:
        # Import here to avoid circular imports
        from main import matcher
        
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
