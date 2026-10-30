from enum import Enum
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class TaxDomain(str, Enum):
    CIT = "cit"
    PROPERTY_TAX = "property_tax"
    GST = "gst"
    STAMP_DUTY = "stamp_duty"
    INDIVIDUAL_TAX = "individual_tax"
    GENERAL = "general"
    OUT_OF_SCOPE = "out_of_scope"


class Intent(str, Enum):
    LOOKUP = "lookup"
    ELIGIBILITY = "eligibility"
    COMPUTATION = "computation"
    DEADLINE = "deadline"
    PROCEDURE = "procedure"
    OUT_OF_SCOPE = "out_of_scope"


# ------------------------------------------------------------------ Domain Entities

class CITEntities(BaseModel):
    """Singapore Corporate Income Tax (CIT) entities and statutory slots."""
    ya: Optional[int] = Field(
        default=None,
        description="Year of Assessment (YA). Format: YYYY (e.g. 2026).")
    fy: Optional[int] = Field(
        default=None,
        description="Financial Year basis period (e.g. FY 2025).")
    fye: Optional[str] = Field(
        default=None,
        description="Financial Year End date in ISO format YYYY-MM-DD (e.g. '2025-12-31').")
    is_first_financial_period: Optional[bool] = Field(
        default=None,
        description="True if first accounts since incorporation cover >12 months (splits into 2 YAs).")
# 2. Financial Figures & Thresholds
    chargeable_income: Optional[float] = Field(
        default=None,
        description="Chargeable Income (taxable profit after tax adjustments & capital allowances).")
    annual_revenue: Optional[float] = Field(
        default=None,
        description="Annual turnover. Key thresholds: <=$200k (Form C-S Lite), <=$5M (Form C-S & ECI waiver).")
# 3. Entity Profile & Exemption Schemes (SUTE vs. PTE)
    is_new_company: Optional[bool] = Field(
        default=None,
        description="True if company is within its first 3 consecutive YAs since incorporation.")
    incorporation_year: Optional[int] = Field(
        default=None,
        description="Year of incorporation in Singapore.")
    tax_resident: Optional[bool] = Field(
        default=None,
        description="True if control & management (Board meetings) are exercised in Singapore.")
    has_individual_shareholder_ge_10pct: Optional[bool] = Field(
        default=None,
        description="SUTE requirement: At least 1 individual holds >=10% of ordinary shares throughout the YA.")
    total_shareholders_count: Optional[int] = Field(
        default=None,
        description="SUTE requirement: Total number of shareholders must be <= 20.")
    is_investment_holding: Optional[bool] = Field(
        default=None,
        description="Disqualifies company from SUTE. Subject to restricted expense deductions under S14D/E.")
    is_property_developer: Optional[bool] = Field(
        default=None,
        description="Disqualifies company from SUTE (developing properties for sale/investment).")
# 4. Filing Form Disqualifiers & Relief Claims
    has_local_employees: Optional[bool] = Field(
        default=None,
        description="Whether company made CPF contributions for local staff (triggers CIT Rebate Cash Grants).")
    company_status: Optional[Literal["active", "dormant"]] = Field(
        default="active",
        description="Dormant companies can apply for filing waiver under Section 13(8) / Form for Dormant Company.")
# 5. Semantic Topics
    topics: List[str] = Field(
        default_factory=list,
        description="IRAS schemes or rules cited (e.g., 'SUTE', 'PTE', 'CIT Rebate', 'Form C-S', 'ECI Waiver').")


class PropertyTaxEntities(BaseModel):
    tax_year: Optional[int] = None
    annual_value: Optional[float] = None
    property_type: Optional[Literal["hdb", "private_residential", "commercial", "industrial"]] = None
    is_owner_occupied: Optional[bool] = None
    topics: List[str] = Field(default_factory=list)


class GSTEntities(BaseModel):
    taxable_turnover: Optional[float] = None
    effective_year: Optional[int] = None
    is_registered: Optional[bool] = None
    topics: List[str] = Field(default_factory=list)


class StampDutyEntities(BaseModel):
    purchase_price_or_market_value: Optional[float] = None
    property_type: Optional[Literal["residential", "non_residential"]] = None
    buyer_profile: Optional[Literal["sc", "pr", "foreigner", "entity"]] = None
    property_count: Optional[int] = None  # relevant for ABSD tiering
    topics: List[str] = Field(default_factory=list)


class IndividualTaxEntities(BaseModel):
    ya: Optional[int] = None
    employment_income: Optional[float] = None
    tax_resident: Optional[bool] = None
    topics: List[str] = Field(default_factory=list)


DomainEntities = Union[
    CITEntities,
    PropertyTaxEntities,
    GSTEntities,
    StampDutyEntities,
    IndividualTaxEntities,
]


# ------------------------------------------------------------------ LLM Input/Output

class SubQueryDraft(BaseModel):
    query: str
    doc_type: Optional[str] = Field(
        default=None,
        description="e.g. rate_table, tax_guide, statutory_rule, faq"
    )


class IntakeOutput(BaseModel):
    """What the intake model extracts: just enough for the deterministic gates (refuse / clarify) to run.
    Query planning is not done here; the supervisor decides whether a question needs sub-queries."""
    domain: TaxDomain
    intent: Intent
    cit: Optional[CITEntities] = None
    property_tax: Optional[PropertyTaxEntities] = None
    gst: Optional[GSTEntities] = None
    stamp_duty: Optional[StampDutyEntities] = None
    individual_tax: Optional[IndividualTaxEntities] = None


class LLMOutput(IntakeOutput):
    """Intake extraction plus sub-queries in the same call (the original pipeline, kept for comparison runs)."""
    sub_queries: List[SubQueryDraft] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """What the query planner (the supervisor's decompose step) returns."""
    sub_queries: List[SubQueryDraft] = Field(default_factory=list)


# ------------------------------------------------------------------ Pipeline Results

class SearchFilter(BaseModel):
    domain: TaxDomain
    year: Optional[int] = None
    # doc_type: Optional[str] = None


class PlannedQuery(BaseModel):
    text: str
    filter: SearchFilter


class Clarify(BaseModel):
    question: str
    missing: List[str]
    domain: TaxDomain
    entities: DomainEntities


class Refuse(BaseModel):
    reason: str


class Decompose(BaseModel):
    domain: TaxDomain
    intent: Intent
    entities: DomainEntities
    tax_year: Optional[int] = None
    assumptions: List[str] = Field(default_factory=list)
    queries: List[PlannedQuery] = Field(default_factory=list)
    # True: `queries` are already a decomposition (written by a planner), so the supervisor searches them as they are.
    # False: `queries` is just the question itself (slim intake); the supervisor decides whether to decompose it.
    planned: bool = True


Result = Union[Clarify, Refuse, Decompose]