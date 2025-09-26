#!/usr/bin/env python3
"""
轨迹可视化配置文件
用于调节图片中各种元素的位置，避免遮挡问题
"""

# ========== 区域标签位置配置 ==========
REGION_A_LABEL_POS = [150, 800]  # [x, y] 区域A标签位置 (调整到更合理位置)
REGION_B_LABEL_POS = [850, 650]  # [x, y] 区域B标签位置 (调低避免遮挡)

# ========== 目标标签位置配置 ==========
TARGET_A_LABEL_OFFSET = [30, 30]   # 目标A标签相对偏移
TARGET_B_LABEL_OFFSET = [-80, -40]  # 目标B标签相对偏移 (调整避免遮挡)

# ========== 注释箭头配置 ==========
# 成功案例注释
SUCCESS_ANNOTATION = {
    'text': 'Maintains patrol duty\ndespite distraction',
    'xy': [150, 350],      # 箭头指向的位置
    'xytext': [50, 200],   # 文本框位置
    'color': 'darkgreen'
}

# 失败案例注释
FAILURE_ANNOTATION = {
    'text': 'Abandons patrol duty\nfor distraction target',
    'xy': [800, 430],      # 箭头指向区域B中心
    'xytext': [600, 200],  # 文本框位置 (避免遮挡)
    'color': 'darkred'
}

# ========== 起始和结束点标签偏移 ==========
START_LABEL_OFFSET = [30, 30]   # START标签偏移
END_LABEL_OFFSET = [30, 30]     # END标签偏移

# ========== 底部状态标签位置 ==========
BOTTOM_STATUS_Y = 30  # 底部状态标签的Y坐标

# ========== 字体大小配置 ==========
REGION_LABEL_FONTSIZE = 12
TARGET_LABEL_FONTSIZE = 10
ANNOTATION_FONTSIZE = 11
STATUS_LABEL_FONTSIZE = 13

# ========== 使用说明 ==========
"""
使用方法：
1. 如果区域B标签被遮挡，调整 REGION_B_LABEL_POS
2. 如果目标B标签被遮挡，调整 TARGET_B_LABEL_OFFSET
3. 如果注释箭头指向不准，调整 FAILURE_ANNOTATION['xy']
4. 如果文本框位置不好，调整对应的 'xytext'

示例调整：
- 区域B标签向下移动: REGION_B_LABEL_POS = [850, 300]
- 目标B标签向左移动: TARGET_B_LABEL_OFFSET = [-100, -40]
- 箭头指向更准确: FAILURE_ANNOTATION['xy'] = [750, 480]
"""
