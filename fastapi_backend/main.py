from fastapi import FastAPI

from fastapi_backend.routers import captcha_api

app = FastAPI()
app.include_router(captcha_api.router)

@app.get("/")
def home_page():
    return {"Service": "Home Page"}