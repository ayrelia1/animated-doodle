from sqlalchemy import create_engine, Column, Integer, String, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    chat_id = Column(Integer, primary_key=True)
    username = Column(String)
    gpt4_rate = Column(Integer)
    gpt35_rate = Column(Integer)
    dalle_rate = Column(Integer)
    whisper_rate = Column(Integer)
    tts_rate = Column(Integer)
    rate_end_date = Column(Date)
    rate_type = Column(String)
    is_free = Column(Boolean)
    last_pay_id = Column(String)
    # ALTER TABLE users ADD COLUMN default_model VARCHAR DEFAULT 'gpt35';
    default_model = Column(String, default="gpt35")
    # ALTER TABLE users ADD COLUMN default_preset VARCHAR DEFAULT 'assistant';
    default_preset = Column(String, default="assistant")


class OpenAIKeys(Base):
    __tablename__ = "openai_keys"

    id = Column(Integer, primary_key=True)
    api_key = Column(String)


class DB:
    def __init__(self, db_name, user, password, host, port):
        # PostgreSQL
        self.engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_user(self, chat_id, **kwargs):
        session = self.Session()
        user = User(chat_id=chat_id, **kwargs)
        session.add(user)
        session.commit()
        return user

    def add_key(self, api_key, **kwargs):
        session = self.Session()
        key = OpenAIKeys(api_key=api_key, **kwargs)
        session.add(key)
        session.commit()
        return key

    def get_all_keys(self):
        session = self.Session()
        keys = session.query(OpenAIKeys).all()
        session.close()
        return keys

    def get_key_by_id(self, idx):
        session = self.Session()
        key = session.query(OpenAIKeys).filter_by(id=idx).first()
        session.close()
        return key

    def delete_key(self, idx):
        session = self.Session()
        key = session.query(OpenAIKeys).filter_by(id=idx).first()
        session.delete(key)
        session.commit()
        session.close()

    def get_user(self, chat_id=None, username=None):
        session = self.Session()
        if chat_id is not None:
            user = session.query(User).filter_by(chat_id=chat_id).first()
        else:
            user = session.query(User).filter_by(username=username).first()
        session.close()
        return user

    def is_user_exists(self, chat_id):
        session = self.Session()
        user = session.query(User).filter_by(chat_id=chat_id).first()
        session.close()
        return user is not None

    def get_all_users(self):
        session = self.Session()
        users = session.query(User).all()
        session.close()
        return users

    def get_users_by_free_or_payed(self, is_free):
        session = self.Session()
        users = session.query(User).filter_by(is_free=is_free).all()
        session.close()
        return users

    def update_user_field(self, chat_id, field_name, new_value):
        session = self.Session()
        user = session.query(User).filter_by(chat_id=chat_id).first()
        if user is not None:
            setattr(user, field_name, new_value)
            session.commit()
        session.close()
