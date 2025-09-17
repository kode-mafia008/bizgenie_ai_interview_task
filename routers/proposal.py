from ai_integration import ProposalRanker
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from data_loader import PROPOSALS_DATA


router = APIRouter()

ranker = ProposalRanker()

class ProposalRequest(BaseModel):
    project_description: str = "Require a Laravel developer with MySQL knowledge to build secure APIs."


@router.post("/rank/")
async def rank_proposals(request: ProposalRequest):
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
