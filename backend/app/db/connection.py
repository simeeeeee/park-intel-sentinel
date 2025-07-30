import os
from mysql.connector import pooling
from dotenv import load_dotenv

# .env 파일 로드 (로컬용, Docker 환경에서는 생략됨)
# env_path = os.path.join(os.path.dirname(__file__), "../../../infra/.env")
# load_dotenv(dotenv_path=env_path)

# 커넥션 설정 (docker-compose 기준 host는 'mariadb' 또는 'db' 사용)
dbconfig = {
    "host": os.getenv("DB_HOST", "db"),  # 기본값 'db'는 docker-compose 서비스 이름
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
}

# 커넥션 풀 생성
connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    pool_reset_session=True,
    **dbconfig
)

def get_connection():
    return connection_pool.get_connection()
