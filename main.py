from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from FastAPI 🚀"}

@app.get("/about")
def about():
    return {"app": "My First FastAPI Project"}
