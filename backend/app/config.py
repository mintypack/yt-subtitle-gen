from pydantic import BaseSettings

class Settings(BaseSettings):
    model_path: str

