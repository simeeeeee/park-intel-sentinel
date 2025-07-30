from connection import get_connection

TABLES = {}

TABLES['car_owners'] = (
    "CREATE TABLE IF NOT EXISTS car_owners ("
    "  id BIGINT NOT NULL PRIMARY KEY,"
    "  name VARCHAR(50),"
    "  phone_number VARCHAR(20),"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME"
    ") ENGINE=InnoDB"
)

TABLES['registered_vehicles'] = (
    "CREATE TABLE IF NOT EXISTS registered_vehicles ("
    "  id BIGINT NOT NULL PRIMARY KEY,"
    "  plate_text VARCHAR(50) NOT NULL UNIQUE,"
    "  owner_id BIGINT NOT NULL,"
    "  is_ev BOOLEAN,"
    "  is_disabled BOOLEAN,"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME,"
    "  FOREIGN KEY (owner_id) REFERENCES car_owners(id)"
    ") ENGINE=InnoDB"
)

TABLES['rfid_tags'] = (
    "CREATE TABLE IF NOT EXISTS rfid_tags ("
    "  id BIGINT NOT NULL PRIMARY KEY,"
    "  rfid_tag VARCHAR(100) NOT NULL UNIQUE,"
    "  created_at DATETIME,"
    "  deleted_at DATETIME"
    ") ENGINE=InnoDB"
)

TABLES['parking_zones'] = (
    "CREATE TABLE IF NOT EXISTS parking_zones ("
    "  id BIGINT NOT NULL PRIMARY KEY,"
    "  name VARCHAR(100) NOT NULL,"
    "  zone_type VARCHAR(100),"
    "  floor INT,"
    "  created_at DATETIME,"
    "  updated_at DATETIME,"
    "  deleted_at DATETIME,"
    "  rfid_tag VARCHAR(100),"
    "  FOREIGN KEY (rfid_tag) REFERENCES rfid_tags(rfid_tag)"
    ") ENGINE=InnoDB"
)

TABLES['robots'] = (
    "CREATE TABLE IF NOT EXISTS robots ("
    "  id BIGINT NOT NULL PRIMARY KEY,"
    "  floor INT,"
    "  created_at DATETIME,"
    "  deleted_at DATETIME,"
    "  updated_at DATETIME"
    ") ENGINE=InnoDB"
)

TABLES['robot_logs'] = (
    "CREATE TABLE IF NOT EXISTS robot_logs ("
    "  id BIGINT NOT NULL PRIMARY KEY,"
    "  zone_id BIGINT NOT NULL,"
    "  robot_id BIGINT NOT NULL,"
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
    "  id BIGINT NOT NULL PRIMARY KEY,"
    "  zone_id BIGINT NOT NULL,"
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
    cursor.close()
    conn.close()
    print("✅ All tables created successfully.")

if __name__ == "__main__":
    create_tables()
