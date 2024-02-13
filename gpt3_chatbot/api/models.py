from sqlalchemy import Column, Integer, String, DateTime
from database import Base

class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    actual_password = Column(String)
    hashed_password = Column(String)
    access_level = Column(Integer) # 0 for admin, 1 for dep head, 2 for user
    department = Column(Integer) # to be assigned