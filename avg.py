import json
import glob
# 初始化字典以累加值和计数
sums = {}
counts = {}
# 步骤 2: 查找所有 JSON 文件的路径
p_values = [0.1, 0.5, 1, 2, 3, 5]
g_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5]

# for p in p_values:
#     for g in g_values:
#         for i in range(1, 5):
#             file_lists = f"/home/hanzhangye/data/proj/ProTAS/results/p_{p}_g_{g}/gtea/epoch50/split_{i}/*.json"
# file_lists = ['/home/hanzhangye/data/proj/ProTAS/results/p_2_g_5e-2/gtea/epoch50/split_{i}/*.json',]
for i in range(1, 5):
    file_paths = glob.glob(f'/home/hanzhangye/data/proj/ProTAS/results/p_2_g_5e-2/gtea/epoch50/split_{i}/*.json', recursive=True)
    # 步骤 3: 遍历每个文件
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                # 累加值
                if key in sums:
                    sums[key] += value
                    counts[key] += 1
                else:
                    sums[key] = value
                    counts[key] = 1

# 步骤 5: 计算平均值
averages = {key: sums[key] / counts[key] for key in sums}

# 步骤 6: 打印结果
for key, value in averages.items():
    print(f"{key}: {value}")