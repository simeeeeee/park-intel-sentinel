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


    now = datetime.now()

    # 1. car_owners
    dummy_data_1 = [
        (1, "김영철", "010-1234-1001", now),
        (2, "이민정", "010-1234-1002", now),
        (3, "박지훈", "010-1234-1003", now),
        (4, "최수연", "010-1234-1004", now),
        (5, "정우성", "010-1234-1005", now),
        (6, "한지민", "010-1234-1006", now),
        (7, "조인성", "010-1234-1007", now),
        (8, "서예진", "010-1234-1008", now),
        (9, "강동원", "010-1234-1009", now),
        (10, "윤소희", "010-1234-1010", now),
    ]

    query = "INSERT INTO car_owners (id, name, phone_number, created_at) VALUES (%s, %s, %s, %s)"
    cursor.executemany(query, dummy_data_1)
    
    
    # 2. registered_vehicles
    plate_texts = [
        '154러7070', '157고4895', '137로2805', '214머4167', '358다6583',
        '368러2704', '242조5916', '230고1494', '173두2632', '110라9081',
        #전기차
        '08호4122', '52주3108', '23가8564', '01서6647'
    ]

    dummy_data_2 = []

    for i in range(1, 15):
        if i > 10:
            vehicle_type = "EV"
        elif i % 2 == 0:
            vehicle_type = "DISABLED"
        else:
            vehicle_type = "NORMAL"

        dummy_data_2.append(
            (i, plate_texts[i-1], (i % 10) + 1, vehicle_type, now)
        )

    query1 = """
    INSERT INTO registered_vehicles 
    (id, plate_text, owner_id, vehicle_type, created_at) 
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor.executemany(query1, dummy_data_2)

    # 3. rfid_tags
    # for i in range(1, 21):
    #     cursor.execute(
    #         "INSERT INTO rfid_tags (id, rfid_tag, created_at) VALUES (%s, %s, %s)",
    #         (i, f"RFID{i:03}", now)
    #     )

    dummy_data_3 = [
        (1, "0643E69B", now),
        (2, "0642919D", now),
        (3, "09BBC4B1", now),
        (4, "0643A29A", now),
        (5, "06441B01", now),
        (6, "0642CD01", now),
        (7, "09491A91", now),
    ]

    query2 = """
    INSERT INTO rfid_tags (id, rfid_tag, created_at)
    VALUES (%s, %s, %s)
    """
    cursor.executemany(query2, dummy_data_3)

    # 4. robots
    for i in range(1, 21):
        cursor.execute(
            "INSERT INTO robots (id, floor, created_at) VALUES (%s, %s, %s)",
            (i, i, now)
        )

    # 5. parking_zones
    rfid_tags = [
        "0643E69B", "0642919D", "09BBC4B1",
        "0643A29A", "06441B01", "0642CD01", "09491A91"
    ]

    # zone_type 그룹별 지정 (4건씩)
    zone_type_map = {
        0: ["EV", "EV", "NORMAL", "NORMAL"],
        1: ["EV", "EV", "NORMAL", "NORMAL"],
        2: ["NONE", "NONE", "NONE", "NONE"],
        3: ["NONE", "NONE", "NONE", "NONE"],
        4: ["DISABLED", "DISABLED", "NORMAL", "NORMAL"],
        5: ["DISABLED", "DISABLED", "NORMAL", "NORMAL"],
        6: ["NONE", "NONE", "NONE", "NONE"],
    }

    for i in range(1, 29):
        group_idx = (i - 1) // 4          # 0~6 (7 groups)
        rfid_tag = rfid_tags[group_idx]
        zone_idx = (i - 1) % 4            # 0~3 (zone 1~4)
        zone_type = zone_type_map[group_idx][zone_idx]
        floor = 1                # 1
        zone_name = f"ZONE{zone_idx + 1}"

        cursor.execute(
            """
            INSERT INTO parking_zones (id, name, zone_type, floor, created_at, rfid_tag)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (i, zone_name, zone_type, floor, now, rfid_tag)
        )

        
    # 6. robot_logs
    # for i in range(1, 21):
    #     cursor.execute(
    #         "INSERT INTO robot_logs (id, zone_id, robot_id, plate_text, rfid_tag, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
    #         (i, i, i, f"차량{i:04}", f"RFID{i:03}", now)
    #     )

    # 7. alert_logs
    # for i in range(1, 21):
    #     cursor.execute(
    #         "INSERT INTO alert_logs (id, zone_id, plate_text, reason, created_at) VALUES (%s, %s, %s, %s, %s)",
    #         (i, i, f"차량{i:04}", "위반 사유 테스트", now)
    #     )

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Dummy data inserted.")

if __name__ == "__main__":
    create_tables()
    insert_dummy_data()
