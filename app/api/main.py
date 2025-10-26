from fastapi import FastAPI
from app.api.routers.scoring import router

app = FastAPI(title="Grant Scoring API (ML)")
app.include_router(router)

@app.get("/")
def health():
    return {"status": "ok"}
