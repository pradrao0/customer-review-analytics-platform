from fastapi import FastAPI


from app.routers.prediction import router as prediction_router


app = FastAPI(
    title="Business Intelligence Platform",
    version="1.0.0"
)

app.include_router(prediction_router)


@app.get("/")
def home():
    return {
        "message": "Business Intelligence Platform API is running."
    }