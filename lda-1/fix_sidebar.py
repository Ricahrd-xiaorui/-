#!/usr/bin/env python
# coding: utf-8

"""
修复sidebar.py中的缩进问题
"""

def fix_indentation():
    try:
        # 读取文件
        with open('modules/sidebar.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并修复缩进问题
        fixed_content = content
        
        # 修复with open后面的st.download_button缩进问题
        pattern = r'(\s+with open\(temp_path, \'r\', encoding=\'utf-8\'\) as f:\s+)(\s*)st\.download_button\('
        replacement = r'\1        st.download_button('
        fixed_content = fixed_content.replace('with open(temp_path, \'r\', encoding=\'utf-8\') as f:\nst.download_button(', 
                                            'with open(temp_path, \'r\', encoding=\'utf-8\') as f:\n        st.download_button(')
        
        # 写回文件
        with open('modules/sidebar.py', 'w', encoding='utf-8') as f:
            f.write(fixed_content)
            
        print("修复完成！")
        return True
    except Exception as e:
        print(f"修复过程中出错: {e}")
        return False

if __name__ == "__main__":
    fix_indentation() 