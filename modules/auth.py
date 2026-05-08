# -*- coding: utf-8 -*-
"""
用户认证模块 - 登录注册功能（MySQL版本）
"""

import streamlit as st
import hashlib
from datetime import datetime
import logging
from utils.db_manager import get_db_manager

logger = logging.getLogger(__name__)


class UserAuth:
    """用户认证管理类（MySQL版本）"""
    
    def __init__(self):
        """初始化用户认证管理器"""
        self.db = get_db_manager()
    
    def _hash_password(self, password):
        """密码哈希加密"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username, password, email=None, require_approval=False):
        """注册新用户"""
        try:
            # 检查用户名格式
            if len(username) < 3:
                return False, "用户名至少3个字符"
            
            if len(username) > 20:
                return False, "用户名最多20个字符"
            
            # 检查密码强度
            if len(password) < 6:
                return False, "密码至少6个字符"
            
            # 检查用户名是否已存在
            sql = "SELECT id FROM users WHERE username = %s"
            existing_user = self.db.execute_one(sql, (username,))
            if existing_user:
                return False, "用户名已存在"
            
            # 创建新用户
            approval_status = 'pending' if require_approval else 'approved'
            sql = """
                INSERT INTO users (username, password, role, email, approval_status, created_at) 
                VALUES (%s, %s, %s, %s, %s, NOW())
            """
            user_id = self.db.execute_insert(
                sql, 
                (username, self._hash_password(password), 'user', email, approval_status)
            )
            
            if user_id:
                logger.info(f"用户注册成功: {username}, 审核状态: {approval_status}")
                if require_approval:
                    return True, "注册成功，请等待管理员审核"
                else:
                    return True, "注册成功"
            else:
                return False, "注册失败，请稍后重试"
                
        except Exception as e:
            logger.error(f"用户注册失败: {e}")
            return False, f"注册失败: {str(e)}"
    
    def login(self, username, password, ip_address=None):
        """用户登录"""
        try:
            # 查询用户
            sql = """
                SELECT id, username, password, role, is_active, approval_status 
                FROM users 
                WHERE username = %s
            """
            user = self.db.execute_one(sql, (username,))
            
            # 检查用户是否存在
            if not user:
                self._log_login(None, username, 'failed', '用户名不存在', ip_address)
                return False, "用户名不存在"
            
            # 检查用户是否被禁用
            if not user['is_active']:
                self._log_login(user['id'], username, 'failed', '账号已被禁用', ip_address)
                return False, "账号已被禁用"
            
            # 检查审核状态
            if user.get('approval_status') == 'pending':
                self._log_login(user['id'], username, 'failed', '账号待审核', ip_address)
                return False, "账号待审核，请联系管理员"
            
            if user.get('approval_status') == 'rejected':
                self._log_login(user['id'], username, 'failed', '账号已拒绝', ip_address)
                return False, "账号已拒绝，请联系管理员"
            
            # 验证密码
            if user['password'] != self._hash_password(password):
                self._log_login(user['id'], username, 'failed', '密码错误', ip_address)
                return False, "密码错误"
            
            # 更新最后登录时间
            sql = "UPDATE users SET last_login = NOW() WHERE id = %s"
            self.db.execute_update(sql, (user['id'],))
            
            # 记录登录成功日志
            self._log_login(user['id'], username, 'success', None, ip_address)
            
            logger.info(f"用户登录成功: {username}")
            return True, "登录成功"
            
        except Exception as e:
            logger.error(f"用户登录失败: {e}")
            return False, f"登录失败: {str(e)}"
    
    def _log_login(self, user_id, username, status, fail_reason=None, ip_address=None):
        """记录登录日志"""
        try:
            sql = """
                INSERT INTO login_logs (user_id, username, login_time, ip_address, status, fail_reason)
                VALUES (%s, %s, NOW(), %s, %s, %s)
            """
            # 如果user_id为None，使用0作为占位符
            self.db.execute_insert(
                sql, 
                (user_id or 0, username, ip_address, status, fail_reason)
            )
        except Exception as e:
            logger.error(f"记录登录日志失败: {e}")
    
    def get_user_info(self, username):
        """获取用户信息"""
        try:
            sql = """
                SELECT id, username, role, email, created_at, last_login, is_active
                FROM users 
                WHERE username = %s
            """
            user = self.db.execute_one(sql, (username,))
            
            if user:
                # 转换datetime为字符串
                if user.get('created_at'):
                    user['created_at'] = user['created_at'].strftime("%Y-%m-%d %H:%M:%S")
                if user.get('last_login'):
                    user['last_login'] = user['last_login'].strftime("%Y-%m-%d %H:%M:%S")
                return user
            return None
            
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None
    
    def change_password(self, username, old_password, new_password):
        """修改密码"""
        try:
            # 查询用户
            sql = "SELECT id, password FROM users WHERE username = %s"
            user = self.db.execute_one(sql, (username,))
            
            if not user:
                return False, "用户不存在"
            
            # 验证旧密码
            if user['password'] != self._hash_password(old_password):
                return False, "原密码错误"
            
            # 检查新密码强度
            if len(new_password) < 6:
                return False, "新密码至少6个字符"
            
            # 更新密码
            sql = "UPDATE users SET password = %s WHERE id = %s"
            self.db.execute_update(sql, (self._hash_password(new_password), user['id']))
            
            logger.info(f"用户修改密码成功: {username}")
            return True, "密码修改成功"
            
        except Exception as e:
            logger.error(f"修改密码失败: {e}")
            return False, f"修改密码失败: {str(e)}"
    
    def get_login_history(self, username, limit=10):
        """获取登录历史"""
        try:
            sql = """
                SELECT login_time, ip_address, status, fail_reason
                FROM login_logs
                WHERE username = %s
                ORDER BY login_time DESC
                LIMIT %s
            """
            logs = self.db.execute_query(sql, (username, limit))
            
            # 转换datetime为字符串
            for log in logs:
                if log.get('login_time'):
                    log['login_time'] = log['login_time'].strftime("%Y-%m-%d %H:%M:%S")
            
            return logs
            
        except Exception as e:
            logger.error(f"获取登录历史失败: {e}")
            return []
    
    def send_verification_code(self, email):
        """发送验证码到邮箱"""
        try:
            import random
            import string
            
            # 生成6位验证码
            code = ''.join(random.choices(string.digits, k=6))
            
            # 存储验证码到session_state（实际应用中应该存储到数据库或Redis）
            import streamlit as st
            if 'verification_codes' not in st.session_state:
                st.session_state.verification_codes = {}
            
            st.session_state.verification_codes[email] = {
                'code': code,
                'timestamp': datetime.now()
            }
            
            # TODO: 实际发送邮件（需要配置SMTP服务器）
            # 这里仅作演示，实际应该使用smtplib发送邮件
            logger.info(f"验证码已生成: {email} -> {code}")
            
            return True, code  # 开发环境返回验证码，生产环境不应返回
            
        except Exception as e:
            logger.error(f"发送验证码失败: {e}")
            return False, None
    
    def verify_code(self, email, code):
        """验证验证码"""
        try:
            import streamlit as st
            
            if 'verification_codes' not in st.session_state:
                return False
            
            if email not in st.session_state.verification_codes:
                return False
            
            stored_data = st.session_state.verification_codes[email]
            stored_code = stored_data['code']
            timestamp = stored_data['timestamp']
            
            # 检查验证码是否过期（5分钟）
            from datetime import timedelta
            if datetime.now() - timestamp > timedelta(minutes=5):
                del st.session_state.verification_codes[email]
                return False
            
            # 验证码匹配
            if stored_code == code:
                del st.session_state.verification_codes[email]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"验证码验证失败: {e}")
            return False
    
    def reset_password_by_email(self, email, new_password):
        """通过邮箱重置密码"""
        try:
            # 查询用户
            sql = "SELECT id, username FROM users WHERE email = %s"
            user = self.db.execute_one(sql, (email,))
            
            if not user:
                return False, "该邮箱未注册"
            
            # 检查新密码强度
            if len(new_password) < 6:
                return False, "新密码至少6个字符"
            
            # 更新密码
            sql = "UPDATE users SET password = %s WHERE id = %s"
            self.db.execute_update(sql, (self._hash_password(new_password), user['id']))
            
            logger.info(f"用户通过邮箱重置密码: {user['username']}")
            return True, "密码重置成功"
            
        except Exception as e:
            logger.error(f"重置密码失败: {e}")
            return False, f"重置密码失败: {str(e)}"
    
    def get_all_users(self, approval_status=None):
        """获取所有用户列表"""
        try:
            sql = "SELECT id, username, role, email, created_at, last_login, is_active, approval_status, permissions FROM users ORDER BY created_at DESC"
            params = []
            if approval_status:
                sql = "SELECT id, username, role, email, created_at, last_login, is_active, approval_status, permissions FROM users WHERE approval_status = %s ORDER BY created_at DESC"
                params = [approval_status]
            users = self.db.execute_query(sql, params)
            return users
        except Exception as e:
            logger.error(f"获取用户列表失败: {e}")
            return []
    
    def approve_user(self, user_id):
        """审核通过用户"""
        try:
            sql = "UPDATE users SET approval_status = 'approved' WHERE id = %s"
            rows = self.db.execute_update(sql, (user_id,))
            if rows > 0:
                logger.info(f"用户审核通过: user_id={user_id}")
                return True, "审核通过"
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"审核用户失败: {e}")
            return False, f"审核失败: {str(e)}"
    
    def reject_user(self, user_id):
        """拒绝用户注册"""
        try:
            sql = "UPDATE users SET approval_status = 'rejected' WHERE id = %s"
            rows = self.db.execute_update(sql, (user_id,))
            if rows > 0:
                logger.info(f"用户被拒绝: user_id={user_id}")
                return True, "已拒绝该用户"
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"拒绝用户失败: {e}")
            return False, f"拒绝失败: {str(e)}"
    
    def reset_user_password(self, user_id, new_password):
        """重置用户密码"""
        try:
            if len(new_password) < 6:
                return False, "密码至少6个字符"
            hashed_pwd = self._hash_password(new_password)
            sql = "UPDATE users SET password = %s WHERE id = %s"
            rows = self.db.execute_update(sql, (hashed_pwd, user_id))
            if rows > 0:
                logger.info(f"用户密码重置成功: user_id={user_id}")
                return True, "密码重置成功"
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"重置密码失败: {e}")
            return False, f"重置失败: {str(e)}"
    
    def update_user_role(self, user_id, role):
        """更新用户角色"""
        try:
            if role not in ['admin', 'user']:
                return False, "无效的角色"
            sql = "UPDATE users SET role = %s WHERE id = %s"
            rows = self.db.execute_update(sql, (role, user_id))
            if rows > 0:
                logger.info(f"用户角色更新: user_id={user_id}, role={role}")
                return True, "角色更新成功"
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"更新角色失败: {e}")
            return False, f"更新失败: {str(e)}"
    
    def toggle_user_active(self, user_id):
        """启用/禁用用户"""
        try:
            sql = "SELECT is_active FROM users WHERE id = %s"
            user = self.db.execute_one(sql, (user_id,))
            if not user:
                return False, "用户不存在"
            new_status = not user['is_active']
            sql = "UPDATE users SET is_active = %s WHERE id = %s"
            self.db.execute_update(sql, (new_status, user_id))
            logger.info(f"用户状态变更: user_id={user_id}, is_active={new_status}")
            return True, "用户状态已更新"
        except Exception as e:
            logger.error(f"变更用户状态失败: {e}")
            return False, f"操作失败: {str(e)}"
    
    def update_user_permissions(self, user_id, permissions):
        """更新用户权限（JSON格式）"""
        try:
            import json
            permissions_json = json.dumps(permissions, ensure_ascii=False)
            sql = "UPDATE users SET permissions = %s WHERE id = %s"
            rows = self.db.execute_update(sql, (permissions_json, user_id))
            if rows > 0:
                logger.info(f"用户权限更新: user_id={user_id}")
                return True, "权限更新成功"
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"更新权限失败: {e}")
            return False, f"更新失败: {str(e)}"
    
    def get_user_permissions(self, user_id):
        """获取用户权限"""
        try:
            import json
            sql = "SELECT permissions FROM users WHERE id = %s"
            user = self.db.execute_one(sql, (user_id,))
            if user and user.get('permissions'):
                return json.loads(user['permissions'])
            return []
        except Exception as e:
            logger.error(f"获取权限失败: {e}")
            return []
    
    def delete_user(self, user_id):
        """删除用户"""
        try:
            sql = "DELETE FROM users WHERE id = %s"
            rows = self.db.execute_update(sql, (user_id,))
            if rows > 0:
                logger.info(f"用户删除: user_id={user_id}")
                return True, "用户已删除"
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"删除用户失败: {e}")
            return False, f"删除失败: {str(e)}"


def render_login_page():
    """渲染登录页面 - 学术专业风格"""
    st.markdown("""
        <style>
        /* 隐藏Streamlit默认元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* 全屏背景 */
        .stApp {
            background: #f8f9fa;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
        }
        
        /* 登录卡片 */
        .login-card {
            background: white;
            border-radius: 12px;
            padding: 24px 40px;
            width: 100%;
            max-width: 630px;
            margin: 0 auto;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid #e8e8e8;
        }
        
        /* Logo和标题区域 */
        .brand-section {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .brand-icon {
            width: 40px;
            height: 40px;
            margin: 0 auto 10px;
            background: linear-gradient(135deg, #1890ff 0%, #096dd9 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .brand-icon svg {
            width: 24px;
            height: 24px;
            fill: white;
        }
        
        .brand-title {
            font-size: 18px;
            font-weight: 600;
            color: #262626;
            margin: 0 0 4px 0;
        }
        
        .brand-subtitle {
            font-size: 12px;
            color: #8c8c8c;
        }
        
        /* 标签页样式 */
        .stTabs {
            margin-bottom: 16px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #fafafa;
            border-radius: 8px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 36px;
            background: transparent;
            border: none;
            border-radius: 6px;
            color: #595959;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s;
            flex: 1;
        }
        
        .stTabs [aria-selected="true"] {
            background: white;
            color: #1890ff;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
        }
        
        /* 输入框标签 */
        .stTextInput > label {
            font-size: 14px;
            font-weight: 500;
            color: #262626;
            margin-bottom: 8px;
        }
        
        /* 输入框样式 */
        .stTextInput > div > div > input {
            border: 1px solid #d9d9d9;
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 14px;
            background: white;
            transition: all 0.2s;
            color: #262626;
            height: 40px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #40a9ff;
            background: white;
            box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.1);
            outline: none;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: #bfbfbf;
        }
        
        /* 按钮样式 */
        .stButton > button {
            width: 100%;
            background: #1890ff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 16px;
            height: 40px;
        }
        
        .stButton > button:hover {
            background: #40a9ff;
        }
        
        .stButton > button:active {
            background: #096dd9;
        }
        
        /* 表单样式 */
        .stForm {
            border: none;
            padding: 0;
        }
        
        /* 消息提示 */
        .stAlert {
            border-radius: 6px;
            border: none;
            padding: 10px 12px;
            font-size: 14px;
            margin-bottom: 16px;
        }
        
        .stSuccess {
            background: #f6ffed;
            color: #52c41a;
            border: 1px solid #b7eb8f;
        }
        
        .stError {
            background: #fff2f0;
            color: #ff4d4f;
            border: 1px solid #ffccc7;
        }
        
        .stInfo {
            background: #e6f7ff;
            color: #1890ff;
            border: 1px solid #91d5ff;
        }
        
        /* 底部链接 */
        .bottom-links {
            text-align: center;
            margin-top: 20px;
            font-size: 13px;
            color: #8c8c8c;
        }
        
        .bottom-links a {
            color: #1890ff;
            text-decoration: none;
        }
        
        .bottom-links a:hover {
            color: #40a9ff;
            text-decoration: underline;
        }
        
        /* 调整容器 */
        .block-container {
            padding-top: 5rem !important;
            padding-bottom: 2rem !important;
            max-width: 670px !important;
        }
        
        /* 隐藏侧边栏 */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* 响应式 */
        @media (max-width: 500px) {
            .login-card {
                padding: 32px 24px;
            }
            
            .block-container {
                padding-top: 3rem !important;
            }
        }
        </style>
        
        <div class="login-card">
            <div class="brand-section">
                <div class="brand-icon">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
                    </svg>
                </div>
                <h1 class="brand-title">政策文本分析系统</h1>
                <p class="brand-subtitle">Policy Text Analysis Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 初始化数据库
    try:
        from utils.db_manager import init_database, test_database_connection
        
        if not test_database_connection():
            st.error("无法连接到数据库")
            st.stop()
        
        if 'db_initialized' not in st.session_state:
            init_database()
            st.session_state.db_initialized = True
            
    except Exception as e:
        st.error(f"数据库初始化失败: {e}")
        st.stop()
    
    auth = UserAuth()
    
    # 标签页
    tab1, tab2, tab3 = st.tabs(["登录", "注册", "找回密码"])
    
    with tab1:
        with st.form("login_form", clear_on_submit=False):
            st.text_input("用户名", key="login_user", placeholder="请输入用户名")
            st.text_input("密码", type="password", key="login_pass", placeholder="请输入密码")
            
            submitted = st.form_submit_button("登录")
            
            if submitted:
                username = st.session_state.login_user
                password = st.session_state.login_pass
                
                if not username or not password:
                    st.error("请输入用户名和密码")
                else:
                    success, message = auth.login(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_info = auth.get_user_info(username)
                        user_perms = auth.get_user_permissions(st.session_state.user_info['id'])
                        st.session_state.user_permissions = user_perms
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    with tab2:
        with st.form("register_form", clear_on_submit=False):
            st.text_input("用户名", key="reg_user", placeholder="3-20个字符")
            st.text_input("邮箱", key="reg_email", placeholder="your@email.com")
            st.text_input("密码", type="password", key="reg_pass", placeholder="至少6个字符")
            st.text_input("确认密码", type="password", key="reg_pass2", placeholder="再次输入密码")
            
            submitted = st.form_submit_button("注册")
            
            if submitted:
                username = st.session_state.reg_user
                email = st.session_state.reg_email
                password = st.session_state.reg_pass
                password2 = st.session_state.reg_pass2
                
                if not username or not email or not password or not password2:
                    st.error("请填写所有字段")
                elif password != password2:
                    st.error("两次密码不一致")
                elif '@' not in email:
                    st.error("邮箱格式不正确")
                else:
                    success, message = auth.register(username, password, email)
                    if success:
                        st.success(f"{message}，请切换到登录页")
                    else:
                        st.error(message)
    
    with tab3:
        render_forgot_password_simple(auth)


def render_forgot_password_simple(auth):
    """渲染忘记密码页面"""
    if 'reset_step' not in st.session_state:
        st.session_state.reset_step = 1
    
    if st.session_state.reset_step == 1:
        with st.form("email_form"):
            st.text_input("邮箱", key="reset_email", placeholder="输入注册邮箱")
            
            submitted = st.form_submit_button("发送验证码")
            
            if submitted:
                email = st.session_state.reset_email
                if not email or '@' not in email:
                    st.error("请输入有效的邮箱")
                else:
                    from utils.db_manager import get_db_manager
                    db = get_db_manager()
                    user = db.execute_one("SELECT username FROM users WHERE email = %s", (email,))
                    
                    if not user:
                        st.error("该邮箱未注册")
                    else:
                        success, code = auth.send_verification_code(email)
                        if success:
                            st.session_state.reset_email_confirmed = email
                            st.session_state.reset_step = 2
                            st.info(f"验证码已发送（开发环境：{code}）")
                            st.rerun()
    
    elif st.session_state.reset_step == 2:
        st.info(f"验证码已发送到：{st.session_state.reset_email_confirmed}")
        
        with st.form("verify_form"):
            st.text_input("验证码", key="verify_code", placeholder="6位数字", max_chars=6)
            st.text_input("新密码", type="password", key="new_pass", placeholder="至少6个字符")
            st.text_input("确认密码", type="password", key="new_pass2", placeholder="再次输入")
            
            col1, col2 = st.columns(2)
            with col1:
                back = st.form_submit_button("返回")
            with col2:
                reset = st.form_submit_button("重置密码")
            
            if back:
                st.session_state.reset_step = 1
                st.rerun()
            
            if reset:
                code = st.session_state.verify_code
                password = st.session_state.new_pass
                password2 = st.session_state.new_pass2
                
                if not code or not password or not password2:
                    st.error("请填写所有字段")
                elif password != password2:
                    st.error("两次密码不一致")
                elif not auth.verify_code(st.session_state.reset_email_confirmed, code):
                    st.error("验证码错误或已过期")
                else:
                    success, message = auth.reset_password_by_email(st.session_state.reset_email_confirmed, password)
                    if success:
                        st.success(f"{message}，请登录")
                        st.session_state.reset_step = 1
                        del st.session_state.reset_email_confirmed
                    else:
                        st.error(message)


def render_user_menu():
    """渲染用户菜单 - 简洁版"""
    if st.session_state.get("logged_in"):
        username = st.session_state.get("username")
        user_info = st.session_state.get("user_info", {})
        
        st.sidebar.markdown("---")
        
        # 用户信息
        role_icon = "👑" if user_info.get("role") == "admin" else "👤"
        st.sidebar.markdown(f"### {role_icon} {username}")
        
        if user_info.get("email"):
            st.sidebar.caption(user_info['email'])
        
        st.sidebar.markdown("")
        
        # 操作按钮
        if st.sidebar.button("🔐 修改密码", use_container_width=True):
            st.session_state.show_change_password = True
        
        if st.sidebar.button("🚪 退出登录", use_container_width=True, type="primary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_info = None
            st.rerun()
        
        # 修改密码对话框
        if st.session_state.get("show_change_password"):
            render_change_password_dialog()


def render_change_password_dialog():
    """渲染修改密码对话框 - 简洁版"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 修改密码")
    
    with st.sidebar.form("change_password_form"):
        st.text_input("原密码", type="password", key="old_pwd")
        st.text_input("新密码", type="password", key="new_pwd")
        st.text_input("确认密码", type="password", key="confirm_pwd")
        
        col1, col2 = st.columns(2)
        with col1:
            cancel = st.form_submit_button("取消")
        with col2:
            submit = st.form_submit_button("确认", type="primary")
        
        if cancel:
            st.session_state.show_change_password = False
            st.rerun()
        
        if submit:
            old_pwd = st.session_state.old_pwd
            new_pwd = st.session_state.new_pwd
            confirm = st.session_state.confirm_pwd
            
            if not old_pwd or not new_pwd or not confirm:
                st.error("请填写所有字段")
            elif new_pwd != confirm:
                st.error("两次密码不一致")
            else:
                auth = UserAuth()
                username = st.session_state.get("username")
                success, message = auth.change_password(username, old_pwd, new_pwd)
                
                if success:
                    st.success(message)
                    st.session_state.show_change_password = False
                    st.rerun()
                else:
                    st.error(message)


def check_authentication():
    """检查用户是否已登录"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        render_login_page()
        st.stop()
    else:
        render_user_menu()


def render_user_management():
    """渲染用户管理界面"""
    import streamlit as st
    import json
    
    auth = UserAuth()
    
    st.markdown("""
        <style>
        .user-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #3498DB;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .status-pending { background: #FFF3CD; border-left-color: #FFC107; }
        .status-approved { background: #D4EDDA; border-left-color: #28A745; }
        .status-rejected { background: #F8D7DA; border-left-color: #DC3545; }
        .status-inactive { opacity: 0.6; }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("👥 用户管理")
    
    # 标签页
    tab1, tab2 = st.tabs(["📋 用户列表", "⏳ 待审核用户"])
    
    with tab1:
        users = auth.get_all_users()
        if users:
            for user in users:
                status_class = f"status-{user['approval_status']}"
                if not user['is_active']:
                    status_class += " status-inactive"
                
                with st.container():
                    st.markdown(f'<div class="user-card {status_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.subheader(f"👤 {user['username']}")
                        st.caption(f"📧 {user['email'] or '未设置邮箱'}")
                    with col2:
                        role_icon = "👑" if user['role'] == 'admin' else "👤"
                        st.markdown(f"**角色**: {role_icon} {user['role']}")
                        status_map = {'pending': '⏳ 待审核', 'approved': '✅ 已通过', 'rejected': '❌ 已拒绝'}
                        st.markdown(f"**状态**: {status_map.get(user['approval_status'], user['approval_status'])}")
                        st.caption(f"创建时间: {user['created_at']}")
                    with col3:
                        if user['username'] != 'admin':  # 不能操作admin账号
                            if st.button(f"⚙️ 管理", key=f"manage_{user['id']}"):
                                st.session_state[f'editing_user_{user["id"]}'] = True
                    
                    # 展开编辑区域
                    if st.session_state.get(f'editing_user_{user["id"]}'):
                        st.divider()
                        st.subheader("用户管理操作")
                        
                        mcol1, mcol2 = st.columns(2)
                        
                        with mcol1:
                            st.markdown("**重置密码**")
                            new_pwd = st.text_input("新密码", type="password", key=f"pwd_{user['id']}")
                            if st.button("🔑 重置密码", key=f"resetpwd_{user['id']}"):
                                if new_pwd:
                                    success, msg = auth.reset_user_password(user['id'], new_pwd)
                                    if success:
                                        st.success(msg)
                                    else:
                                        st.error(msg)
                                else:
                                    st.warning("请输入新密码")
                        
                        with mcol2:
                            st.markdown("**角色设置**")
                            new_role = st.selectbox("选择角色", ['user', 'admin'], index=0 if user['role'] == 'user' else 1, key=f"role_{user['id']}")
                            if st.button("👑 更新角色", key=f"updaterole_{user['id']}"):
                                success, msg = auth.update_user_role(user['id'], new_role)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                        st.divider()
                        
                        # 状态操作
                        acol1, acol2, acol3 = st.columns(3)
                        with acol1:
                            status_text = "🔓 启用" if not user['is_active'] else "🔒 禁用"
                            if st.button(status_text, key=f"toggle_{user['id']}"):
                                success, msg = auth.toggle_user_active(user['id'])
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                        with acol2:
                            if st.button("🗑️ 删除用户", key=f"delete_{user['id']}", type="primary"):
                                if st.session_state.get(f"confirm_delete_{user['id']}"):
                                    success, msg = auth.delete_user(user['id'])
                                    if success:
                                        st.success(msg)
                                        st.rerun()
                                    else:
                                        st.error(msg)
                                else:
                                    st.session_state[f"confirm_delete_{user['id']}"] = True
                                    st.warning("再次点击确认删除")
                        
                        with acol3:
                            if st.button("✖️ 关闭", key=f"close_{user['id']}"):
                                st.session_state[f'editing_user_{user["id"]}'] = False
                                st.rerun()
                        
                        # 权限设置
                        st.divider()
                        st.markdown("**🔐 权限设置**")
                        current_perms = auth.get_user_permissions(user['id'])
                        available_perms = [
                            'data_load', 'text_process', 'basic_analysis', 
                            'topic_modeling', 'visualization', 'export', 'advanced_analysis'
                        ]
                        perm_labels = {
                            'data_load': '📂 数据加载',
                            'text_process': '✏️ 文本预处理',
                            'basic_analysis': '📈 基础分析',
                            'topic_modeling': '🧠 主题建模',
                            'visualization': '🎨 可视化',
                            'export': '📦 结果导出',
                            'advanced_analysis': '🔬 高级分析'
                        }
                        
                        selected_perms = []
                        pcols = st.columns(3)
                        for idx, perm in enumerate(available_perms):
                            with pcols[idx % 3]:
                                if st.checkbox(perm_labels[perm], value=perm in current_perms, key=f"perm_{user['id']}_{perm}"):
                                    selected_perms.append(perm)
                        
                        if st.button("💾 保存权限", key=f"saveperm_{user['id']}"):
                            success, msg = auth.update_user_permissions(user['id'], selected_perms)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("暂无用户数据")
    
    with tab2:
        pending_users = auth.get_all_users(approval_status='pending')
        if pending_users:
            st.subheader(f"待审核用户 ({len(pending_users)})")
            for user in pending_users:
                with st.container():
                    st.markdown(f'<div class="user-card status-pending">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.subheader(f"👤 {user['username']}")
                        st.caption(f"📧 {user['email'] or '未设置'}")
                        st.caption(f"申请时间: {user['created_at']}")
                    with col2:
                        if st.button("✅ 通过", key=f"approve_{user['id']}", type="primary"):
                            success, msg = auth.approve_user(user['id'])
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col3:
                        if st.button("❌ 拒绝", key=f"reject_{user['id']}"):
                            success, msg = auth.reject_user(user['id'])
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 没有待审核的用户")


def require_auth(func):
    """装饰器：要求用户登录"""
    def wrapper(*args, **kwargs):
        check_authentication()
        return func(*args, **kwargs)
    return wrapper


def check_permission(permission):
    """检查用户是否有指定权限"""
    if not st.session_state.get('logged_in'):
        return False

    # 管理员拥有所有权限
    user_info = st.session_state.get('user_info', {})
    if user_info.get('role') == 'admin':
        return True

    # 检查用户权限列表
    user_perms = st.session_state.get('user_permissions', [])
    return permission in user_perms


def require_permission(permission):
    """装饰器：要求用户有指定权限"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_permission(permission):
                st.error('您没有访问此功能的权限')
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator
