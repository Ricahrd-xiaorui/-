# -*- coding: utf-8 -*-
"""
数据库配置模块
"""

import os
from pathlib import Path

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'kongruiqi',  # ← 改成你的MySQL root密码
    'database': 'policy_analysis',
    'charset': 'utf8mb4',
    'autocommit': True
}

# 数据库初始化SQL
INIT_SQL = """
-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS policy_analysis
DEFAULT CHARACTER SET utf8mb4
DEFAULT COLLATE utf8mb4_unicode_ci;

USE policy_analysis;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(64) NOT NULL COMMENT 'SHA256加密',
    role ENUM('admin', 'user') DEFAULT 'user',
    email VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    approval_status ENUM('pending', 'approved', 'rejected') DEFAULT 'approved',
    permissions TEXT COMMENT 'JSON格式存储权限列表',
    INDEX idx_username (username),
    INDEX idx_created_at (created_at),
    INDEX idx_approval_status (approval_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

-- 登录日志表
CREATE TABLE IF NOT EXISTS login_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    username VARCHAR(50) NOT NULL,
    login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(50),
    user_agent TEXT,
    status ENUM('success', 'failed') DEFAULT 'success',
    fail_reason VARCHAR(200),
    INDEX idx_user_id (user_id),
    INDEX idx_login_time (login_time),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='登录日志表';

-- 用户会话表
CREATE TABLE IF NOT EXISTS user_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    session_id VARCHAR(100) NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_session_id (session_id),
    INDEX idx_user_id (user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户会话表';

-- 插入默认管理员账号（如果不存在）
INSERT IGNORE INTO users (username, password, role, approval_status, created_at)
VALUES ('admin', 'ce3f8b781f2a40eb73b7ccf084be7ad7b65205abe8b335798f2b871e5969023e', 'admin', 'approved', NOW());
-- 密码: kongruiqi (SHA256)
"""
