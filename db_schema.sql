-- 1. DB 생성
CREATE DATABASE smartfarm_db DEFAULT CHARACTER SET utf8mb4;

-- 2. (필수) DB 사용 선언
USE smartfarm_db;

-- 3. 'Chambers' 테이블 생성
CREATE TABLE Chambers (
    chamber_id INT AUTO_INCREMENT PRIMARY KEY,
    chamber_no INT NOT NULL UNIQUE
);

-- 4. 'Pigs' 테이블 생성
CREATE TABLE Pigs (
    pig_id INT AUTO_INCREMENT PRIMARY KEY,
    pig_no INT NOT NULL,
    pig_class VARCHAR(50),
    chamber_id INT,
    UNIQUE (chamber_id, pig_no),
    FOREIGN KEY (chamber_id) REFERENCES Chambers(chamber_id)
);

-- 5. 'Pig_Logs' (완전한 스키마) 생성
CREATE TABLE Pig_Logs (
    pig_log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    pig_id INT,
    timestamp DATETIME NOT NULL,
    weight_kg DECIMAL(6, 2),
    breath_rate INT,
    temp_rectal DECIMAL(4, 1),
    temp_back DECIMAL(4, 1),
    temp_neck DECIMAL(4, 1),
    temp_head DECIMAL(4, 1),
    manure_g DECIMAL(8, 2),
    sensible_heat DECIMAL(8, 2),
    latent_heat DECIMAL(8, 2),
    annotations JSON,
    FOREIGN KEY (pig_id) REFERENCES Pigs(pig_id)
);

-- 6. 'Chamber_Logs' (완전한 스키마) 생성
CREATE TABLE Chamber_Logs (
    chamber_log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chamber_id INT,
    timestamp DATETIME NOT NULL,
    temperature DECIMAL(4, 1),
    humidity DECIMAL(4, 1),
    co2 DECIMAL(6, 1),
    nh3 DECIMAL(4, 1),
    ventilation_rate DECIMAL(5, 2),
    feed_volume DECIMAL(6, 3),
    water_supply DECIMAL(5, 1),
    UNIQUE (chamber_id, timestamp),
    FOREIGN KEY (chamber_id) REFERENCES Chambers(chamber_id)
);

-- 7. 'Users' 테이블 생성
CREATE TABLE Users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    user_role VARCHAR(20) NOT NULL DEFAULT 'viewer',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 8. 'Equipment_Logs' 테이블 생성
CREATE TABLE Equipment_Logs (
    equip_log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chamber_id INT NOT NULL,
    timestamp DATETIME NOT NULL,
    equipment_type VARCHAR(50) NOT NULL,
    status ENUM('ON', 'OFF', 'ERROR', 'STANDBY') NOT NULL,
    power_usage_wh DECIMAL(8, 2),
    FOREIGN KEY (chamber_id) REFERENCES Chambers(chamber_id),
    UNIQUE (chamber_id, equipment_type, timestamp)
);