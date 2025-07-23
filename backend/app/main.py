from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Intelligence Cargo System backend"}

@app.get("/health")
def health_check():
    return {"status": "ok"}