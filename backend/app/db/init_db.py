import os
import random
from datetime import datetime
from mysql.connector import pooling
from dotenv import load_dotenv

load_dotenv()

# DB 커넥션
dbconfig = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "password"),
    "database": os.getenv("MYSQL_DATABASE", "test_db"),
}
connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    pool_reset_session=True,
    **dbconfig
)
def get_connection():
    return connection_pool.get_connection()

# 테이블 정의
TABLES = {}
TABLES['car_owners'] = (
    "CREATE TABLE IF NOT EXISTS car_owners ("
    "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
    "  name VARCHAR(50),"
    "  phone_number VARCHAR(20),"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME"
    ") ENGINE=InnoDB"
)

TABLES['registered_vehicles'] = (
    "CREATE TABLE IF NOT EXISTS registered_vehicles ("
    "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
    "  plate_text VARCHAR(50) NOT NULL UNIQUE,"
    "  owner_id BIGINT,"
    "  vehicle_type VARCHAR(20),"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME,"
    "  FOREIGN KEY (owner_id) REFERENCES car_owners(id)"
    ") ENGINE=InnoDB"
)

TABLES['rfid_tags'] = (
    "CREATE TABLE IF NOT EXISTS rfid_tags ("
    "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
    "  rfid_tag VARCHAR(100) NOT NULL UNIQUE,"
    "  created_at DATETIME,"
    "  deleted_at DATETIME"
    ") ENGINE=InnoDB"
)

TABLES['robots'] = (
    "CREATE TABLE IF NOT EXISTS robots ("
    "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
    "  floor INT,"
    "  created_at DATETIME,"
    "  deleted_at DATETIME,"
    "  updated_at DATETIME"
    ") ENGINE=InnoDB"
)

TABLES['parking_zones'] = (
    "CREATE TABLE IF NOT EXISTS parking_zones ("
    "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
    "  name VARCHAR(100),"
    "  zone_type VARCHAR(100),"
    "  floor INT,"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME,"
    "  rfid_tag VARCHAR(100),"
    "  FOREIGN KEY (rfid_tag) REFERENCES rfid_tags(rfid_tag)"
    ") ENGINE=InnoDB"
)

TABLES['robot_logs'] = (
    "CREATE TABLE IF NOT EXISTS robot_logs ("
    "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
    "  zone_id BIGINT,"
    "  robot_id BIGINT,"
    "  plate_text VARCHAR(50),"
    "  rfid_tag VARCHAR(100),"
    "  image_path TEXT,"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME,"
    "  FOREIGN KEY (zone_id) REFERENCES parking_zones(id),"
    "  FOREIGN KEY (robot_id) REFERENCES robots(id),"
    "  FOREIGN KEY (plate_text) REFERENCES registered_vehicles(plate_text),"
    "  FOREIGN KEY (rfid_tag) REFERENCES rfid_tags(rfid_tag)"
    ") ENGINE=InnoDB"
)

TABLES['alert_logs'] = (
    "CREATE TABLE IF NOT EXISTS alert_logs ("
    "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
    "  zone_id BIGINT,"
    "  plate_text VARCHAR(50),"
    "  reason TEXT,"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME,"
    "  FOREIGN KEY (zone_id) REFERENCES parking_zones(id),"
    "  FOREIGN KEY (plate_text) REFERENCES registered_vehicles(plate_text)"
    ") ENGINE=InnoDB"
)

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    for name, ddl in TABLES.items():
        print(f"📦 Creating table `{name}`...")
        cursor.execute(ddl)
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ All tables created.")

def insert_dummy_data():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_connection()
    cursor = conn.cursor()

    # 1. car_owners
    for i in range(1, 21):
        cursor.execute(
            "INSERT INTO car_owners (id, name, phone_number, created_at) VALUES (%s, %s, %s, %s)",
            (i, f"Owner{i}", f"010-1234-{1000+i}", now)
        )

    # 2. registered_vehicles
    for i in range(1, 21):
        if i % 3 == 0:
            vehicle_type = "DISABLED"
        elif i % 2 == 0:
            vehicle_type = "EV"
        else:
            vehicle_type = "NORMAL"

        cursor.execute(
            "INSERT INTO registered_vehicles (id, plate_text, owner_id, vehicle_type, created_at) VALUES (%s, %s, %s, %s, %s)",
            (i, f"차량{i:04}", i, vehicle_type, now)
        )

    # 3. rfid_tags
    for i in range(1, 21):
        cursor.execute(
            "INSERT INTO rfid_tags (id, rfid_tag, created_at) VALUES (%s, %s, %s)",
            (i, f"RFID{i:03}", now)
        )

    # 4. robots
    for i in range(1, 21):
        cursor.execute(
            "INSERT INTO robots (id, floor, created_at) VALUES (%s, %s, %s)",
            (i, i % 5, now)
        )

    # 5. parking_zones
    for i in range(1, 21):
        if i % 3 == 0:
            zone_type = "DISABLED"
        elif i % 2 == 0:
            zone_type = "EV"
        else:
            zone_type = "NORMAL"
        cursor.execute(
            "INSERT INTO parking_zones (id, name, zone_type, floor, created_at, rfid_tag) VALUES (%s, %s, %s, %s, %s, %s)",
            (i, f"ZONE{i}", zone_type, i % 5, now, f"RFID{i:03}")
        )

    # 6. robot_logs
    for i in range(1, 21):
        cursor.execute(
            "INSERT INTO robot_logs (id, zone_id, robot_id, plate_text, rfid_tag, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (i, i, i, f"차량{i:04}", f"RFID{i:03}", now)
        )

    # 7. alert_logs
    for i in range(1, 21):
        cursor.execute(
            "INSERT INTO alert_logs (id, zone_id, plate_text, reason, created_at) VALUES (%s, %s, %s, %s, %s)",
            (i, i, f"차량{i:04}", "위반 사유 테스트", now)
        )

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Dummy data inserted.")

if __name__ == "__main__":
    create_tables()
    insert_dummy_data()
