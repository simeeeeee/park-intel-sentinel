from dotenv import load_dotenv
import os

# 도커 컨테이너 내부 절대경로 기준
load_dotenv(dotenv_path="/app/infra/.env")

from databases import Database

DATABASE_URL = (
    f"mysql+aiomysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 3306)}/{os.getenv('MYSQL_DATABASE')}"
)

database = Database(DATABASE_URL)
