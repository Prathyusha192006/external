import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI Churn Inference Server on http://127.0.0.1:8000 ...")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("Interactive Web Dashboard: http://127.0.0.1:8000/dashboard")
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
