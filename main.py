from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your symptom router
# from apps.symptoms.routes import routes as symptom_router

app = FastAPI(
    title="Medical Diagnosis API",
    description="API to manage cases and symptoms for diagnosis models",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
    openapi_url="/openapi.json"
)

# Optional: CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    # Add your frontend domain if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your routers
# app.include_router(symptom_router, prefix="/symptoms", tags=["Symptoms"])
