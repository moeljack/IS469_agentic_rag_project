"""Data contracts for Step 1 (Intake) and Step 2 (Planner).

Two layers:
  * LLMOutput  - what the small model is asked to produce (lenient, all optional)
  * Plan / Clarify / Refuse - what the rest of the pipeline consumes (strict, built by code)
"""
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


class Intent(str, Enum):
    LOOKUP = "lookup"              # what is the rate / rule / definition
    ELIGIBILITY = "eligibility"    # does the company qualify for X
    COMPUTATION = "computation"    # compute tax payable / exemption / rebate
    DEADLINE = "deadline"          # filing or payment dates
    PROCEDURE = "procedure"        # how to file / claim / apply
    OUT_OF_SCOPE = "out_of_scope"


class DocType(str, Enum):
    ACT = "act"                    # Income Tax Act (tier 1)
    ETAX_GUIDE = "etax_guide"      # IRAS e-Tax Guides (tier 2)
    IRAS_PAGE = "iras_page"        # IRAS web pages (tier 2)
    ANNOUNCEMENT = "announcement"  # Budget / MOF / IRAS announcements (tier 3)


class Entities(BaseModel):
    """Only what the user explicitly said. None = unknown. Never guessed."""
    ya: Optional[int] = Field(None, description="Year of Assessment, only if the user says YA / Year of Assessment")
    fy: Optional[int] = Field(None, description="Financial year, only if the user says FY / financial year")
    fye: Optional[str] = Field(None, description="Financial year end date as YYYY-MM-DD")
    is_new_company: Optional[bool] = Field(None, description="True if the user says the company is newly incorporated")
    incorporation_year: Optional[int] = None
    chargeable_income: Optional[float] = Field(None, description="Chargeable income in SGD")
    tax_resident: Optional[bool] = None
    topics: list[str] = Field(default_factory=list,
                              description="Official scheme / form names, e.g. 'Start-Up Tax Exemption', 'Form C-S'")


# class SubQuery(BaseModel):
#     query: str = Field(description="Keyword-style retrieval query using official IRAS terminology")
#     doc_types: list[DocType] = Field(default_factory=list,
#                                      description="Preferred document types; empty = any")


class LLMOutput(BaseModel):
    intent: Intent
    entities: Entities = Field(default_factory=Entities)


# ---------------------------------------------------------------- pipeline outputs

class SearchFilter(BaseModel):
    """Applied by the retrieval step. `ya` should match chunks where
    ya_from <= ya <= ya_to, OR where the chunk has no YA bounds."""
    ya: Optional[int] = None
    doc_types: list[DocType] = Field(default_factory=list)
    max_tier: int = 3
    only_current: bool = True      # exclude chunks with superseded_by set


class PlannedQuery(BaseModel):
    text: str
    filter: SearchFilter


class Decompose(BaseModel):
    intent: Intent
    entities: Entities
    ya: int                        # resolved Year of Assessment
    assumptions: list[str] = Field(default_factory=list)   # surfaced to the user in the answer
    # tools: list[str] = Field(default_factory=list)         # deterministic tools Step 3 should call


class Clarify(BaseModel):
    question: str
    missing: list[str]
    entities: Entities             # carry these into the next turn


class Refuse(BaseModel):
    reason: str


Result = Union[Decompose, Clarify, Refuse]