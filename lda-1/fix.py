#!/usr/bin/env python
# coding: utf-8

"""
修复model_trainer.py中的缩进问题
"""

def fix_indentation():
    try:
        # 读取文件
        with open('modules/model_trainer.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找问题行
        for i, line in enumerate(lines):
            if 'else:' in line and i+2 < len(lines) and 'self.model = LdaModel(' in lines[i+2]:
                # 找到了问题行，修复缩进
                print(f"找到问题行: {i+1}, {line.strip()}")
                print(f"问题行下一行: {i+2}, {lines[i+1].strip()}")
                print(f"问题行下两行: {i+3}, {lines[i+2].strip()}")
                
                # 修复缩进
                if not lines[i+2].startswith('            '):
                    lines[i+2] = '            ' + lines[i+2].lstrip()
                    print("已修复缩进")
                    
                    # 检查后续行的缩进
                    j = i + 3
                    while j < len(lines) and ('corpus=' in lines[j] or 
                                             'id2word=' in lines[j] or 
                                             'num_topics=' in lines[j] or
                                             'iterations=' in lines[j] or
                                             'passes=' in lines[j] or
                                             'chunksize=' in lines[j] or
                                             'alpha=' in lines[j] or
                                             'eta=' in lines[j] or
                                             'eval_every=' in lines[j] or
                                             'random_state=' in lines[j] or
                                             'callbacks=' in lines[j]):
                        if not lines[j].startswith('                '):
                            lines[j] = '                ' + lines[j].lstrip()
                            print(f"修复了第{j+1}行的缩进")
                        j += 1
        
        # 写回文件
        with open('modules/model_trainer.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        print("修复完成！")
        return True
    except Exception as e:
        print(f"修复过程中出错: {e}")
        return False

if __name__ == "__main__":
    fix_indentation() 