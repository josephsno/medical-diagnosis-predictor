import math
from typing import TypeVar, Generic, List, Optional, Any
from fastapi import Query, Request
from pydantic import BaseModel

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    count: int
    current_page: Optional[int]
    next: Optional[str]
    previous: Optional[str]
    per_page: int
    total_page: int
    hasItem: bool
    results: List[T]  # ← was List[Any], change to List[T]

    class Config:
        from_attributes = True  # ← add this too

class CustomPagination:
    def __init__(
        self,
        per_page: int = Query(default=10, alias="per_page"),
        page: int = Query(default=1, ge=1),
    ):
        self.max_page_size = 1000
        self.per_page = min(per_page, self.max_page_size) if per_page > 0 else 10
        self.page = page
        self.offset = (page - 1) * self.per_page

    def paginate(self, data: list, total: int, request: Request, total_base: int = 1) -> dict:
        total_page = math.ceil(total / self.per_page) if self.per_page else 1
        base_url = str(request.base_url) + request.url.path.lstrip("/")
        
        # Build next/previous links
        def build_url(p):
            return f"{base_url}?page={p}&per_page={self.per_page}"

        next_link = build_url(self.page + 1) if self.page < total_page else None
        prev_link = build_url(self.page - 1) if self.page > 1 else None

        return {
            "count": total,
            "current_page": self.page,
            "next": next_link,
            "previous": prev_link,
            "per_page": self.per_page,
            "total_page": total_page,
            "hasItem": total_base > 0,
            "results": data,
        }