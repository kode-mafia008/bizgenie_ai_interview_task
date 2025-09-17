import os 
import pandas as pd 


FREELANCERS_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "freelancers.csv"))
PROJECTS_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "projects.csv"))
PROPOSALS_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "proposals.csv"))
