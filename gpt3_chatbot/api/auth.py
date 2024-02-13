from datetime import timedelta, datetime
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException, Response, Cookie
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette import status
from database import SessionLocal
from models import Users
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt
from dotenv import load_dotenv
from diskcache import Cache
import os

# ----- constants --------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
SECRET_REFRESH_KEY = os.getenv("SECRET_REFRESH_KEY")
ALGORITHM = os.getenv("ALGORITHM")
TOKEN_EXPIRE_MINUTES = 20
REFRESH_TOKEN_EXPIRE_MINUTES = 60
# ---------------------------


# Create cache instance for blacklisted tokens
blacklist_cache = Cache(directory="token_blacklist")

router = APIRouter(
    prefix = "/auth",
    tags = ['auth']
)

bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token') # endpoint

class CreateUserRequest(BaseModel):
    username: str
    password: str
    department: int

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    refresh_exp: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]

@router.post("/register",status_code=status.HTTP_201_CREATED)
def create_user(db: db_dependency, create_user_request: CreateUserRequest):
    create_user_model = Users(
        username=create_user_request.username,
        actual_password = create_user_request.password,
        hashed_password=bcrypt_context.hash(create_user_request.password),
        access_level=2, # Access level 2 by default
        department=create_user_request.department
    )

    db.add(create_user_model)
    db.commit()

    return {"message":"User created successfully"}

@router.post("/token", response_model=Token)
def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency, response: Response):    
    user = authenticate_user(form_data.username, form_data.password, db)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user')
    # creating tokens
    access_token = create_access_token(user.username, user.id, user.access_level, user.department, timedelta(minutes=TOKEN_EXPIRE_MINUTES))
    print("uname", user.username)
    print("user.id", user.access_level)
    print("udept", user.department)
    print("tdelta", timedelta(minutes=TOKEN_EXPIRE_MINUTES))
    print("access tokeen", access_token)
    
    
    
    
    
    refresh_token = create_refresh_token(user.username, timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES))

    # Store refresh token in cookies, for docs: https://www.starlette.io/responses/#set-cookie
    response.set_cookie(key='refresh_token', value=refresh_token, 
                        expires=REFRESH_TOKEN_EXPIRE_MINUTES * 60, path='/', 
                        domain=None, secure=False, httponly=True, samesite='lax')
    
    response.set_cookie('access_token', access_token, TOKEN_EXPIRE_MINUTES * 60,
                        TOKEN_EXPIRE_MINUTES * 60, '/', None, False, True, 'lax')
    
    response.set_cookie('logged_in', 'True', TOKEN_EXPIRE_MINUTES * 60,
                        TOKEN_EXPIRE_MINUTES * 60, '/', None, False, False, 'lax')
    
    refresh_payload = jwt.decode(refresh_token, SECRET_REFRESH_KEY, algorithms=[ALGORITHM])
    refresh_exp = refresh_payload.get("exp")
    print("refresh_token",refresh_token)
    # print("refresh_payload",refresh_payload)
    # print(datetime.utcfromtimestamp(refresh_exp))
    # print(type(datetime.utcfromtimestamp(refresh_exp)))
    return {'refresh_token': refresh_token, 'access_token': access_token, 'token_type':'bearer', 'refresh_exp': refresh_exp}


def authenticate_user(username:str, password:str, db):
    user = db.query(Users).filter(Users.username == username).first()
    if not user:
        return False
    if not bcrypt_context.verify(password, user.hashed_password):
        return False
    return user

def get_user_from_db(username: str, db):
    user = db.query(Users).filter(Users.username == username).first()
    if not user:
        return False
    return user

# Generate jwt using python jose
def create_access_token(username: str, user_id: int, access_level: int, department: int, expires_delta: timedelta):
    encode = {'sub':username, 'id':user_id, 'access':access_level, 'dept':department}
    expires = datetime.utcnow() + expires_delta
    encode.update({'exp':expires})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)

# Generate refresh token to allow generation of access token if user is logged in
def create_refresh_token(username: str, expires_delta: timedelta):
    encode = {'username':username}
    expires = datetime.utcnow() + expires_delta
    encode.update({'exp':expires})
    return jwt.encode(encode, SECRET_REFRESH_KEY, algorithm=ALGORITHM)


# Add token to blacklist to prevent use of access tokens
def blacklist_token(token: str, expires: int):
    blacklist_cache.set(key=token, value=expires, expire=TOKEN_EXPIRE_MINUTES*60)
    print(blacklist_cache)
    return True

# Check if token is blacklisted
def is_token_blacklisted(token: str):
    if blacklist_cache.get(token) is None:
        return False
    return True

def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    if is_token_blacklisted(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get('sub')
        user_id: int = payload.get('id')
        access_level: int = payload.get('access')
        department: int = payload.get('dept')
        if username is None or user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user.')
        return {'username': username, 'id': user_id, 'access':access_level, 'dept':department}
    except:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate the user')
    
# Add access_token to blacklist and deletes access_token and logged_in cookie values
@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(response: Response, token: Annotated[str, Depends(oauth2_bearer)]):
    if token is None:
        raise HTTPException(status_code=401, detail="Authentication Failed")
    expires = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]).get('exp')
    blacklist_token(token, expires)
    response.set_cookie(key = 'logged_in', value='')
    response.set_cookie(key = 'access_token', value='')
    return {'status': 'success'}

# Generates a new access token if current access token and refresh token is not yet expired
@router.post("/refresh")
def refresh(access_token:str, refresh_token: str, db: db_dependency, response:Response):
    try:
        access_payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        access_exp = access_payload.get("exp")
        refresh_payload = jwt.decode(refresh_token, SECRET_REFRESH_KEY, algorithms=[ALGORITHM])
        refresh_exp = refresh_payload.get("exp")
        username = refresh_payload.get("username")
        if (datetime.utcfromtimestamp(refresh_exp) > datetime.utcnow()) and (datetime.utcfromtimestamp(access_exp) > datetime.utcnow()):
            blacklist_token(access_token, access_exp)
            user = get_user_from_db(username, db)
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user///')
            access_token = create_access_token(username, user.id, user.access_level, user.department, timedelta(minutes=TOKEN_EXPIRE_MINUTES))
            # change access token and logged in cookie 
            response.set_cookie('access_token', access_token, TOKEN_EXPIRE_MINUTES * 60,
                        TOKEN_EXPIRE_MINUTES * 60, '/', None, False, True, 'lax')
    
            response.set_cookie('logged_in', 'True', TOKEN_EXPIRE_MINUTES * 60,
                        TOKEN_EXPIRE_MINUTES * 60, '/', None, False, False, 'lax')
            
            return {"access_token":access_token, 'token_type':'bearer'}
        # else:
            # print("refresh_exp", refresh_exp)
            # print("datetime.utcfromtimestamp(refresh_exp)", datetime.utcfromtimestamp(refresh_exp))
            # print("datetime.utcnow()", datetime.utcnow())

    except:
        # print("refresh_exp", refresh_exp)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user')