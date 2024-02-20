from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

# ---- constants ----
load_dotenv()

user = os.getenv("PG_USERNAME")
pword = os.getenv("PG_PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")
dbname = os.getenv("DATABASE")

URL_DATABASE = f'postgresql://{user}:{pword}@{host}:{port}/{dbname}'

engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()