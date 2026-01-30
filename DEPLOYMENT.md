# 部署指南 / Deployment Guide

本文档说明如何将系统部署到Streamlit Cloud，特别是如何解决中文字体显示问题。

This document explains how to deploy the system to Streamlit Cloud, especially how to solve Chinese font display issues.

---

## 📋 部署前准备 / Pre-deployment Checklist

### 1. 下载中文字体

**重要**：在部署前必须先下载字体文件到 `fonts/` 目录，并提交到Git仓库。

**方法A：使用自动脚本**
```bash
python download_font.py
```

**方法B：手动下载**
1. 访问 https://github.com/notofonts/noto-cjk/tree/main/Sans/OTF/SimplifiedChinese
2. 下载 `NotoSansCJKsc-Regular.otf` 文件（约16-20MB）
3. 将文件保存到项目的 `fonts/` 目录

### 2. 验证文件结构

确保以下文件存在：
```
项目根目录/
├── fonts/
│   ├── NotoSansCJKsc-Regular.otf  ← 必须存在！
│   └── README.md
├── packages.txt                    ← 必须存在！
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── app.py
```

### 3. 提交到Git仓库

```bash
# 添加字体文件
git add fonts/NotoSansCJKsc-Regular.otf
git add fonts/README.md

# 添加配置文件
git add packages.txt
git add .streamlit/config.toml

# 提交
git commit -m "添加中文字体支持和部署配置"

# 推送到远程仓库
git push origin main
```

---

## 🚀 Streamlit Cloud 部署步骤

### 1. 登录 Streamlit Cloud

访问 https://share.streamlit.io/ 并使用GitHub账号登录。

### 2. 创建新应用

1. 点击 "New app" 按钮
2. 选择你的GitHub仓库
3. 选择分支（通常是 `main` 或 `master`）
4. 设置主文件路径为 `app.py`
5. 点击 "Deploy" 开始部署

### 3. 等待部署完成

首次部署需要：
- 安装Python依赖（requirements.txt）
- 安装系统包（packages.txt）
- 启动应用

整个过程可能需要5-10分钟。

### 4. 验证字体加载

部署完成后：
1. 访问应用URL
2. 上传测试文档并训练模型
3. 在"主题可视化"标签页生成词云
4. 检查词云是否正确显示中文

---

## 🔍 故障排除 / Troubleshooting

### 问题1：词云仍然显示方框

**可能原因**：
- 字体文件未提交到Git仓库
- 字体文件路径不正确
- 字体文件损坏或不完整

**解决方案**：
1. 检查GitHub仓库中是否存在 `fonts/NotoSansCJKsc-Regular.otf`
2. 检查文件大小是否正常（应该是16-20MB）
3. 重新下载字体文件并提交
4. 在Streamlit Cloud上点击 "Reboot app"

### 问题2：部署失败，提示找不到包

**可能原因**：
- `requirements.txt` 或 `packages.txt` 文件缺失或格式错误

**解决方案**：
1. 检查 `requirements.txt` 文件是否存在且格式正确
2. 检查 `packages.txt` 文件是否存在
3. 查看部署日志中的具体错误信息
4. 修复后重新提交并部署

### 问题3：应用运行缓慢

**可能原因**：
- Streamlit Cloud免费版资源有限
- 处理大量文档或大文件

**解决方案**：
1. 减少上传的文档数量
2. 降低主题数量
3. 考虑升级到Streamlit Cloud付费版
4. 或自行部署到其他云服务器

### 问题4：查看部署日志

在Streamlit Cloud应用页面：
1. 点击右下角的 "Manage app"
2. 选择 "Logs" 标签
3. 查看实时日志输出
4. 搜索 "字体" 或 "font" 关键词查看字体加载情况

---

## 📊 字体加载机制说明

系统按以下优先级查找字体：

1. **项目字体目录**（优先级最高）
   - `fonts/NotoSansSC-Regular.otf`
   - `fonts/NotoSansCJKsc-Regular.otf`
   - `fonts/SourceHanSansSC-Regular.otf`
   - `fonts/simhei.ttf`
   - `fonts/msyh.ttc`

2. **matplotlib字体管理器**
   - 自动检测系统已安装的中文字体

3. **系统字体路径**
   - Windows: `C:\Windows\Fonts\`
   - macOS: `/System/Library/Fonts/`
   - Linux: `/usr/share/fonts/`

4. **packages.txt安装的字体**
   - `fonts-noto-cjk`
   - `fonts-wqy-zenhei`
   - `fonts-wqy-microhei`

如果找到字体，日志中会显示：
```
使用项目字体: fonts/NotoSansCJKsc-Regular.otf
```

如果未找到字体，日志中会显示：
```
未找到中文字体，词云可能无法正确显示中文
```

---

## 🌐 其他部署选项

### Docker 部署

如果需要使用Docker部署，可以创建以下Dockerfile：

```dockerfile
FROM python:3.9-slim

# 安装系统依赖和中文字体
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 暴露端口
EXPOSE 8501

# 启动应用
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

构建和运行：
```bash
docker build -t policy-analysis .
docker run -p 8501:8501 policy-analysis
```

### 云服务器部署

在云服务器（如阿里云、腾讯云、AWS等）上部署：

1. **安装依赖**
```bash
# 更新系统
sudo apt-get update

# 安装Python和pip
sudo apt-get install python3 python3-pip

# 安装中文字体
sudo apt-get install fonts-noto-cjk fonts-wqy-zenhei

# 克隆项目
git clone [你的仓库URL]
cd [项目目录]

# 安装Python依赖
pip3 install -r requirements.txt
```

2. **运行应用**
```bash
# 前台运行
streamlit run app.py

# 后台运行（使用nohup）
nohup streamlit run app.py --server.port=8501 &

# 或使用screen
screen -S streamlit
streamlit run app.py
# 按 Ctrl+A+D 退出screen
```

3. **配置反向代理（可选）**

使用Nginx配置反向代理：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## 📞 获取帮助

如果遇到部署问题：

1. 查看本文档的故障排除部分
2. 查看 [fonts/README.md](fonts/README.md) 了解字体详情
3. 查看Streamlit Cloud部署日志
4. 在GitHub仓库提交Issue

---

## ✅ 部署检查清单

部署前请确认：

- [ ] 已下载字体文件到 `fonts/` 目录
- [ ] 字体文件大小正常（16-20MB）
- [ ] `packages.txt` 文件存在
- [ ] `requirements.txt` 文件完整
- [ ] `.streamlit/config.toml` 文件存在
- [ ] 所有文件已提交到Git仓库
- [ ] 已推送到GitHub远程仓库
- [ ] 在Streamlit Cloud上创建了应用
- [ ] 部署完成后测试了词云功能
- [ ] 词云能够正确显示中文

---

## 📝 更新日志

- 2026-01-30: 添加中文字体支持和部署配置
- 2026-01-30: 创建部署指南文档
