from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


Route = Literal["auto_resolve", "escalate", "refuse", "ask_more_info"]


class ClassificationOutput(BaseModel):
    category: Literal["technical", "billing", "general", "account", "feature_request"]
    key_issues: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PolicyOutput(BaseModel):
    policy_found: bool
    refund_related: bool = False
    refund_abuse: bool = False
    abusive_content_request: bool = False
    policy_violation: bool = False
    relevant_document_ids: list[int] = Field(default_factory=list)
    reason: str


class TriageOutput(BaseModel):
    action: Route
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class GroundingOutput(BaseModel):
    groundedness_score: float = Field(ge=0.0, le=1.0)
    answer_confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    needs_more_retrieval: bool = False


class QueryRefinement(BaseModel):
    refined_query: str


class RevisedResponse(BaseModel):
    response: str
