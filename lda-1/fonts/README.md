# 字体文件说明 / Font Files

## 用途 / Purpose

此目录用于存放中文字体文件，以确保在Streamlit Cloud等Linux服务器上能够正确显示词云中的中文字符。

This directory is for storing Chinese font files to ensure proper display of Chinese characters in word clouds on Linux servers like Streamlit Cloud.

---

## 下载字体 / Download Fonts

### 方法1：从GitHub下载 Noto Sans CJK SC（推荐）

**Noto Sans CJK SC** 是Google和Adobe联合开发的开源中文字体，使用SIL Open Font License授权。

1. 访问GitHub仓库：
   https://github.com/notofonts/noto-cjk

2. 下载字体文件（选择以下任一方式）：

   **方式A：直接下载单个文件**
   - 访问：https://github.com/notofonts/noto-cjk/tree/main/Sans/OTF/SimplifiedChinese
   - 下载 `NotoSansCJKsc-Regular.otf` 文件
   - 将文件放入本目录（`fonts/`）

   **方式B：使用wget命令**
   ```bash
   cd fonts
   wget https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
   ```

   **方式C：使用curl命令**
   ```bash
   cd fonts
   curl -L -o NotoSansCJKsc-Regular.otf https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
   ```

3. 确认文件已下载到 `fonts/NotoSansCJKsc-Regular.otf`

### 方法2：从Releases下载完整字体包

1. 访问：https://github.com/notofonts/noto-cjk/releases
2. 下载最新版本的 `Sans.zip`
3. 解压后找到 `Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf`
4. 将文件复制到本目录

### 方法3：使用其他开源中文字体

如果无法下载Noto字体，也可以使用其他开源中文字体：

- **思源黑体（Source Han Sans）**：https://github.com/adobe-fonts/source-han-sans
- **文泉驿微米黑**：http://wenq.org/wqy2/index.cgi?MicroHei

---

## 验证安装 / Verify Installation

下载字体后，重新运行应用，系统会自动检测并使用项目字体目录中的字体文件。

After downloading the font, restart the application. The system will automatically detect and use the font files in the project fonts directory.

查看日志输出，应该能看到类似以下信息：
```
使用项目字体: fonts/NotoSansCJKsc-Regular.otf
```

---

## 支持的字体文件 / Supported Font Files

系统会按以下优先级查找字体文件：

1. `fonts/NotoSansSC-Regular.otf`
2. `fonts/NotoSansCJKsc-Regular.otf`
3. `fonts/SourceHanSansSC-Regular.otf`
4. `fonts/simhei.ttf`
5. `fonts/msyh.ttc`

只需下载其中任意一个即可。

---

## 许可证 / License

- **Noto Sans CJK**: SIL Open Font License 1.1
- **Source Han Sans**: SIL Open Font License 1.1
- **文泉驿微米黑**: GPL v3 with font embedding exception

这些字体都是开源字体，可以自由使用、修改和分发。

---

## 故障排除 / Troubleshooting

### 问题：词云仍然显示方框

**解决方案：**

1. 确认字体文件已正确下载到 `fonts/` 目录
2. 检查文件名是否正确（区分大小写）
3. 确认文件大小不为0（完整下载）
4. 重启Streamlit应用
5. 查看应用日志中的字体检测信息

### 问题：Streamlit Cloud部署后仍无法显示中文

**解决方案：**

1. 确保 `fonts/` 目录和字体文件已提交到Git仓库
2. 确保 `packages.txt` 文件存在于项目根目录
3. 在Streamlit Cloud上重新部署应用
4. 等待部署完成后测试

---

## 文件大小参考 / File Size Reference

- NotoSansCJKsc-Regular.otf: 约 16-20 MB
- SourceHanSansSC-Regular.otf: 约 16-20 MB
- wqy-microhei.ttc: 约 4-5 MB

如果文件大小明显偏小，可能下载不完整，请重新下载。
