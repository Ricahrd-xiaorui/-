#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体下载脚本 / Font Download Script

自动下载Noto Sans CJK SC字体文件到fonts目录
Automatically download Noto Sans CJK SC font file to fonts directory
"""

import os
import sys
import urllib.request
from pathlib import Path

# 字体下载URL（使用多个备用源）
FONT_URLS = [
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
]
FONT_FILENAME = "NotoSansCJKsc-Regular.otf"
FONTS_DIR = "fonts"

def download_font():
    """下载字体文件"""
    # 创建fonts目录
    fonts_path = Path(FONTS_DIR)
    fonts_path.mkdir(exist_ok=True)
    
    # 目标文件路径
    font_file = fonts_path / FONT_FILENAME
    
    # 检查文件是否已存在
    if font_file.exists():
        file_size = font_file.stat().st_size
        if file_size > 1_000_000:  # 大于1MB，认为是有效文件
            print(f"✓ 字体文件已存在: {font_file}")
            print(f"  文件大小: {file_size / 1_048_576:.2f} MB")
            return True
        else:
            print(f"⚠ 字体文件不完整，重新下载...")
            font_file.unlink()
    
    print(f"正在下载字体文件...")
    print()
    
    # 尝试多个下载源
    for i, url in enumerate(FONT_URLS, 1):
        print(f"尝试下载源 {i}/{len(FONT_URLS)}")
        print(f"URL: {url}")
        
        try:
            # 设置请求头，模拟浏览器
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            # 下载文件
            def report_progress(block_num, block_size, total_size):
                """显示下载进度"""
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded / total_size * 100, 100)
                    mb_downloaded = downloaded / 1_048_576
                    mb_total = total_size / 1_048_576
                    
                    # 使用\r实现进度条覆盖
                    sys.stdout.write(f"\r下载进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, font_file, reporthook=report_progress)
            print()  # 换行
            
            # 验证下载
            file_size = font_file.stat().st_size
            if file_size > 1_000_000:
                print(f"✓ 字体下载成功!")
                print(f"  文件路径: {font_file}")
                print(f"  文件大小: {file_size / 1_048_576:.2f} MB")
                return True
            else:
                print(f"✗ 下载失败: 文件大小异常 ({file_size} bytes)")
                font_file.unlink()
                
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            if font_file.exists():
                font_file.unlink()
            print()
    
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("Noto Sans CJK SC 字体下载工具")
    print("=" * 60)
    print()
    
    success = download_font()
    
    print()
    if success:
        print("=" * 60)
        print("字体安装完成！")
        print("现在可以运行 streamlit run app.py 启动应用")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("所有下载源均失败！")
        print()
        print("请尝试手动下载：")
        print(f"1. 访问: https://github.com/notofonts/noto-cjk/tree/main/Sans/OTF/SimplifiedChinese")
        print(f"2. 点击 NotoSansCJKsc-Regular.otf 文件")
        print(f"3. 点击 'Download' 按钮下载")
        print(f"4. 将下载的文件保存到: {FONTS_DIR}/{FONT_FILENAME}")
        print()
        print("或查看 fonts/README.md 了解其他下载方式")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
