# BizGenie AI Engineer Technical Test - Complete Solution

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel

# =============================================================================
# Data Models
# =============================================================================
 

@dataclass
class RankedProposal:
    proposal_id: str
    freelancer_id: str
    final_score: float
    relevance_score: float
    rating: float
    bid_price: int
    success_rate: int

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
