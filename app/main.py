from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routers.analyze import router as analyze_router

app = FastAPI()

# Static & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Routers
app.include_router(analyze_router, prefix="/api")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
