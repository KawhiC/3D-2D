import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import sys
import subprocess

# 设置页面标题
st.title("医学图像分割任务页面")

uploaded_file = st.file_uploader("上传一张图片", type=["png", "jpg", "jpeg"])
json_file_path = "./data_demo/label2image_test.json"
with open(json_file_path, 'w') as f:
    json.dump({}, f)

if uploaded_file is not None:
    # 加载图片
    image = Image.open(uploaded_file)
    file_path = os.path.join("./data_demo/images", uploaded_file.name)
    # 加载当前图片（原始图像或上一次预测的结果）
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    target_width = 700
    original_width, original_height = image.size
    new_height = int((target_width / original_width) * original_height)
    image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
    img_width, img_height = image.size  # PIL 的 size 返回 (宽, 高)
    img_ratio = img_width / img_height  # 宽高比
    max_canvas_width = target_width  # 画布的最大宽度
    if img_width > max_canvas_width:  # 如果图片宽度超出最大宽度
        canvas_width = max_canvas_width
        canvas_height = int(canvas_width / img_ratio)  # 按比例缩放高度
    else:
        canvas_width = img_width
        canvas_height = img_height

    # 设置绘图区域
    st.write("在图片上框选矩形方框：")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # 默认填充颜色
        stroke_width=2,  # 边框宽度
        stroke_color="rgba(255, 0, 0, 1)",  # 边框颜色
        background_image=image,  # 设置背景图片
        update_streamlit=True,
        height=canvas_height,  # 动态设置画布高度
        width=canvas_width,  # 动态设置画布宽度
        drawing_mode="rect",  # 绘制模式：矩形
        key="canvas",  # 唯一标识
    )
        # 检查绘制结果
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        print(len(objects))
        if len(objects) > 0:
            print(st.session_state.idx)
            # 继续后续处理
            original_file_name = uploaded_file.name.split(".")[0]
            # 创建一个占位符
            image_placeholder = st.empty()
            obj = objects[len(objects)-1]
            left = int(obj['left'] * original_width / target_width)
            top = int(obj['top'] * original_height / new_height)
            width = int(obj['width'] * original_width / target_width)
            height = int(obj['height'] * original_height / new_height)
            # 创建掩码图像
            mask_image = Image.new("L", (original_width, original_height), 0)
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle([left, top, left + width, top + height], fill=255)
            mask_file_name = f"./data_demo/masks/{original_file_name}_mask_{st.session_state.idx}.png"
            mask_image.save(mask_file_name)
            # 更新 JSON 文件
            with open(json_file_path, 'w') as f:
                json.dump({}, f)
            with open(json_file_path, 'r') as f:
                content = json.load(f)
                content[mask_file_name[2:]] = f"data_demo/images/{original_file_name}.png"
            with open(json_file_path, 'w') as f_new:
                json.dump(content, f_new, indent=4)           
            # 执行预测任务
                command = [
                sys.executable,
                "./test.py",
                 "--save_pred", "True",
                 "--sam_checkpoint", "workdir/models/sam-med2d/epoch15_sam.pth",
            ]
            with st.spinner("正在处理，请稍候..."):
                subprocess.run(command)
            # 可视化叠加结果
            # 加载原始图像
            original_photo = f"data_demo/images/{original_file_name}.png"
            original_image = Image.open(original_photo).convert("RGBA")  # 转为 RGBA 模式
            # 初始化透明叠加图层
            combined_image = original_image.copy()
            # 依次加载所有掩码图像并叠加
            for idx in range(len(objects)):  # 根据对象数量循环
                generate_mask = f"./workdir/sammed/boxes_prompt/{original_file_name}_mask_{idx}.png"
                mask_image = Image.open(generate_mask).convert("L")  # 转为灰度图

                # 创建掩码叠加图层
                mask_overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))  # 创建透明图层
                for x in range(mask_image.size[0]):
                    for y in range(mask_image.size[1]):
                        if mask_image.getpixel((x, y)) > 0:  # 如果掩码的像素值大于0，表示该位置需要叠加
                            mask_overlay.putpixel((x, y), (255, 255, 255, 128))  # 叠加半透明白色
                # 将掩码叠加到原始图像上
                combined_image = Image.alpha_composite(combined_image, mask_overlay)
            combined_image_path = f"./output/{original_file_name}_combined.png"
            combined_image.save(combined_image_path)
            
            original_width, original_height = combined_image.size
            new_height = int((target_width / original_width) * original_height)
            combined_image = combined_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
            img_width, img_height = combined_image.size  # PIL 的 size 返回 (宽, 高)
            img_ratio = img_width / img_height  # 宽高比
            max_canvas_width = target_width  # 画布的最大宽度
            if img_width > max_canvas_width:  # 如果图片宽度超出最大宽度
                canvas_width = max_canvas_width
                canvas_height = int(canvas_width / img_ratio)  # 按比例缩放高度
            else:
                canvas_width = img_width
                canvas_height = img_height
            st.session_state.idx+=1
            image_placeholder.image(combined_image, caption="分割结果", use_container_width=False)
else:
    st.write("请上传一张图片以继续操作。") 
