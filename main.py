
from fastapi import FastAPI
from ai_integration import FreelancerMatcher, ProposalRanker
from routers import match, proposal
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging

# =============================================================================
# Part 3: FastAPI Microservice
# =============================================================================

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


# Initialize components
app = FastAPI(title="BizGenie AI Matching Service", version="1.0.0")

matcher = FreelancerMatcher()
ranker = ProposalRanker()


@app.on_event("startup")
async def startup_event():
    """Initialize the matching system on startup.""" 
    logging.info("BizGenie AI Matching Service started successfully!")

# Include routers
app.include_router(prefix="/match", router=match.router)
app.include_router(prefix="/proposal", router=proposal.router)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
