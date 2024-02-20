from fastapi import FastAPI, status, HTTPException, Depends
from typing import Annotated
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import models
import auth
from auth import get_current_user

app = FastAPI()
app.include_router(auth.router)

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]

# Need to login first to access this endpoint
@app.get("/user", status_code=status.HTTP_200_OK)
def user(user:user_dependency, db:db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication Failed")
    return {"User":user}

# For Access Level 1 or 0 only
@app.get("/admin", status_code=status.HTTP_200_OK)
def for_admins_only(user:user_dependency, db:db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication Failed")
    if user["access"] not in (1,0):
        raise HTTPException(status_code=401, detail="Access denied")
    return {"User":user, "is_admin":True}

@app.get("/dep_one", status_code=status.HTTP_200_OK)
def for_dep_one_only(user:user_dependency, db:db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication Failed")
    if user["dept"] != (1):
        raise HTTPException(status_code=401, detail="Access denied. User's department is not valid.")
    return {"User":user, "valid_dept":True}
