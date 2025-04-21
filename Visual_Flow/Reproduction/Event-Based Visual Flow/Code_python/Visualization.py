import cv2  # 引入 OpenCV 库，用于图像处理
import numpy as np  # 引入 NumPy 库，用于数值计算

# Channel-by-channel accumulation
# 将 events 按时间戳累积到不同通道内
def gen_events_array(events_ori, C_event, duration, event_h, event_w):
    events = np.zeros((event_h, event_w, C_event))  # 初始化一个 event_h x event_w x C_event 的零张量，用于存储累积事件
    C_inter = duration / C_event  # 计算每个时间通道对应的时间间隔
    for i in range(events_ori.shape[0]):  # 遍历所有原始事件
        t, W, H, p = events_ori[i]  # 解包当前事件的 (时间戳, 列, 行,  极性)
        p = -1 if p == 0 else p  # 将极性 p 从 {0,1} 映射到 {-1,1}
        channel_idx = min(int(t // C_inter), C_event - 1)  # 计算事件所属的时间通道索引，确保不超出范围
        events[int(H), int(W), channel_idx] += p  # 在对应通道的位置累积极性值
    return events  # 返回累积后的事件数组

# 可视化事件函数
# 根据灰度图生成 RGB 显示图，将正负极事件映射到不同颜色
def vis_event(gray):
    h, w = gray.shape  # 获取灰度图的高度和宽度
    out = 255 * np.ones([h, w, 3])  # 创建一个全白 RGB 图像作为背景
    pos_weight = gray.copy()  # 复制灰度图，用于计算正极性权重
    neg_weight = gray.copy()  # 复制灰度图，用于计算负极性权重

    pos_weight[pos_weight < 0] = 0  # 将负值设为 0，只保留正值
    pos_weight = pos_weight * 2 * 255  # 放大正极性权重到 [0, 510]

    neg_weight[neg_weight > 0] = 0  # 将正值设为 0，只保留负值
    neg_weight = np.abs(neg_weight) * 2 * 255  # 放大负极性权重到 [0, 510]

    out[..., 1] = out[..., 1] - pos_weight - neg_weight  # 在绿色通道减去正负权重值
    out[..., 0] -= pos_weight  # 在蓝色通道减去正极性权重
    out[..., 2] -= neg_weight  # 在红色通道减去负极性权重
    out = out.clip(0, 255)  # 限制像素值到 [0, 255]
    return out.astype(np.uint8)  # 转换为 uint8 类型并返回


def read_events_from_file(file_path):
    """
    Read event data from a text file.
    
    Each line should be in the format: [x, y, t, polarity]

    Parameters:
        file_path (str): Path to the text file

    Returns:
        events (np.ndarray): N×3 NumPy array, each row is [x, y, t, polarity]
    """
    events = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Remove surrounding square brackets if present
            if line.startswith('[') and line.endswith(']'):
                line = line[1:-1]
            # Split by comma
            parts = line.split(' ')
            if len(parts) >= 4:
                try:
                    t = float(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    polarity = float(parts[3].strip())  
                    events.append([t, x, y, polarity])
                except ValueError:
                    # Skip the line if conversion fails
                    continue
    return np.array(events)


# 主函数入口
if __name__ == "__main__":
    file_path = r"Visual_Flow\Reproduction\Event-Based Visual Flow\Test_Dataset\IROS_Dataset-2018-independent-motion\IROS_Dataset\multiple_objects\2_objs\events.txt"  # Event data file, each line is in the format [x,y,t,polarity]
    events_ori = read_events_from_file(file_path) # 加载原始事件数据，格式为 N x 4 (T,W,H,P)
    event_w = int(np.max(events_ori[:, 1]) + 1)  # 计算事件图像的宽度（最大列索引 + 1）
    event_h = int(np.max(events_ori[:, 2]) + 1)  # 计算事件图像的高度（最大行索引 + 1）
    C_event = 5  # 设置时间通道数

    events_ori[:, 0] = events_ori[:, 0] - events_ori[:, 0].min()  # 将时间戳平移到从 0 开始
    events_ori[:, 3] = (events_ori[:, 3] - 0.5) * 2  # 将极性 P 从 {0,1} 映射到 {-1,1}
    duration = events_ori[:, 0].max() - events_ori[:, 0].min()  # 计算时间戳的持续时长
    print("duration:", duration)  # 打印持续时长

    events1 = gen_events_array(events_ori, C_event, duration, event_h, event_w)  # 生成累积事件数组
    events1_vis = events1.clip(-C_event, C_event) / (2 * C_event)  # 限制累积值到 [-5, 5] 并归一化到 [-0.5, 0.5]

    cv2.imwrite('Visual_Flow\Reproduction\Event-Based Visual Flow\Output\events1_vis.jpg', vis_event(events1_vis[..., 2]))  # 可视化并保存第 3 个通道的事件图像
