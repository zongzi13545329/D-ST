import os
import shutil

# 设置根目录
root_dir = "/home/lin01231/public/datasets/image_crop2/image_crop2"

# 遍历所有jpg文件
for fname in os.listdir(root_dir):
    if fname.endswith(".jpg") and "-" in fname:
        try:
            # 拆分文件名
            patient_id, tracking = fname.replace(".jpg", "").split("-")
            tracking = tracking.strip()
            
            # 构造新文件夹路径
            patient_dir = os.path.join(root_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # 原始文件路径
            old_path = os.path.join(root_dir, fname)
            # 新文件路径
            new_path = os.path.join(patient_dir, fname)

            # 移动文件
            shutil.move(old_path, new_path)
        except Exception as e:
            print(f"跳过文件 {fname}，错误：{e}")
