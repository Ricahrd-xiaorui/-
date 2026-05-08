# -*- coding: utf-8 -*-
"""
数据库连接管理模块
"""

import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
import logging
from config.database import DB_CONFIG, INIT_SQL

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, config=None):
        """初始化数据库管理器"""
        self.config = config or DB_CONFIG
        self._connection = None

    def get_connection(self):
        """获取数据库连接"""
        try:
            if self._connection is None or not self._connection.open:
                self._connection = pymysql.connect(
                    host=self.config['host'],
                    port=self.config['port'],
                    user=self.config['user'],
                    password=self.config['password'],
                    database=self.config['database'],
                    charset=self.config['charset'],
                    cursorclass=DictCursor,
                    autocommit=self.config.get('autocommit', True)
                )
            return self._connection
        except pymysql.Error as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None

    @contextmanager
    def get_cursor(self):
        """获取游标（上下文管理器）"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute_query(self, sql, params=None):
        """执行查询（SELECT）"""
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchall()

    def execute_one(self, sql, params=None):
        """执行查询并返回单条记录"""
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchone()

    def execute_update(self, sql, params=None):
        """执行更新（INSERT/UPDATE/DELETE）"""
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.rowcount

    def execute_insert(self, sql, params=None):
        """执行插入并返回插入的ID"""
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.lastrowid

    def init_database(self):
        """初始化数据库（创建表和默认数据）"""
        try:
            temp_config = self.config.copy()
            temp_config.pop('database', None)

            conn = pymysql.connect(
                host=temp_config['host'],
                port=temp_config['port'],
                user=temp_config['user'],
                password=temp_config['password'],
                charset=temp_config['charset']
            )

            cursor = conn.cursor()

            for statement in INIT_SQL.split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        logger.warning(f"执行SQL语句失败: {statement[:50]}... 错误: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            self._add_missing_columns()

            logger.info("数据库初始化成功")
            return True
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False

    def _add_missing_columns(self):
        """检查并添加缺失的字段（兼容旧版本MySQL）"""
        try:
            result = self.execute_one("""
                SELECT 1 FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'users' AND COLUMN_NAME = 'approval_status'
            """, (self.config['database'],))

            if not result:
                logger.info("添加 approval_status 字段...")
                self.execute_update("""
                    ALTER TABLE users ADD COLUMN approval_status ENUM('pending', 'approved', 'rejected') DEFAULT 'approved'
                """)

            result = self.execute_one("""
                SELECT 1 FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'users' AND COLUMN_NAME = 'permissions'
            """, (self.config['database'],))

            if not result:
                logger.info("添加 permissions 字段...")
                self.execute_update("""
                    ALTER TABLE users ADD COLUMN permissions TEXT COMMENT 'JSON格式存储权限列表'
                """)

            result = self.execute_one("""
                SELECT 1 FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'users' AND INDEX_NAME = 'idx_approval_status'
            """, (self.config['database'],))

            if not result:
                logger.info("添加 idx_approval_status 索引...")
                self.execute_update("""
                    ALTER TABLE users ADD INDEX idx_approval_status (approval_status)
                """)

            logger.info("字段检查完成")
        except Exception as e:
            logger.error(f"添加缺失字段失败: {e}")

    def test_connection(self):
        """测试数据库连接"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False


_db_manager = None


def get_db_manager():
    """获取全局数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database():
    """初始化数据库"""
    db = get_db_manager()
    return db.init_database()


def test_database_connection():
    """测试数据库连接"""
    db = get_db_manager()
    return db.test_connection()
