from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    person_id: int | None = None                       # use a stored synthetic person
    features: dict[str, float] | None = None           # or the observed features directly


class RecourseRequest(BaseModel):
    person_id: int
    horizon: int = Field(12, ge=1, le=24)
    target_prob: float = Field(0.8, gt=0, lt=1)
    rho_override: float | None = Field(None, ge=0, le=0.95)


class NoLabelRequest(BaseModel):
    proxy_table: list[dict]
    approval_by_cell: list[dict]
