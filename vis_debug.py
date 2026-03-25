import torch

# 替换成你生成的新 info_3d.pth 和原版的 info_3d.pth 路径
new_info_path = "/data/ZhaoX/OVM3D-Det-1/pseudo_label/SUNRGBD/train/info_3d.pth"
old_info_path = "/data/ZhaoX/OVM3D-Det-1/pseudo_label/SUNRGBD/train/info_3d_original.pth" # 如果有备份的话

def analyze_info(path, name):
    info = torch.load(path)
    total_boxes = 0
    valid_boxes = 0
    invalid_boxes = 0
    
    for im_id, data in info.items():
        if 'boxes3d' not in data: continue
        for box in data['boxes3d']:
            total_boxes += 1
            if box[0][0] == -1: # 检测是不是被我们填了 -1
                invalid_boxes += 1
            else:
                valid_boxes += 1
                
    print(f"=== {name} 诊断报告 ===")
    print(f"总目标数: {total_boxes}")
    print(f"有效 3D 框: {valid_boxes}")
    print(f"被丢弃(无效)框: {invalid_boxes}")
    print(f"丢弃率: {invalid_boxes / total_boxes * 100:.2f}%\n")

analyze_info(new_info_path, "DetAny3D 提取版")
# analyze_info(old_info_path, "OVM3D 原版") # 如果你有备份可以取消注释对比