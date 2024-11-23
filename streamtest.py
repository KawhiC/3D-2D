import streamlit as st
import subprocess
import numpy as np
import os
import sys
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import json
# 设置网页标题和布局

st.set_page_config(page_title="实时医学图像分割", layout="wide", initial_sidebar_state="expanded")

# 侧边栏设置
st.sidebar.title("导航")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "选择功能",
    options=["首页", "3D医学图像分割", "2D医学图像分割", "关于"],
)

# 首页内容
if page == "首页":
    st.title("✨ 医学图像分割平台 ✨")
    st.markdown(
        """
        <style>
        .big-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .info-box {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class='info-box'>
            <p class="big-title">欢迎来到医学图像分割平台！</p>
            <ul>
                <li>🚀 提供先进的 <strong>3D医学图像分割</strong> 和 <strong>2D医学图像分割</strong> 功能。</li>
                <li>📁 支持上传 npz 和常见通用图像格式。</li>
                <li>🧠 使用最新的深度学习算法，提供高精度的分割结果。</li>
                <li>🔍 提供结果可视化功能，轻松查看分割效果。</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(
        "show.png",
        caption="分割示例",
        use_container_width=True,
    )

    st.markdown("### 平台亮点")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("💡 <strong>高效分割</strong><br>轻松应对复杂场景。", unsafe_allow_html=True)

    with col2:
        st.markdown("🎨 <strong>简单易用</strong><br>为医生和普通人设计。", unsafe_allow_html=True)

    with col3:
        st.markdown("🌐 <strong>多器官支持</strong><br>覆盖肺部、脑部等重要领域。", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 快速开始")

    # 调整列比例，保证更好的布局
    col1, col2 = st.columns([3, 2])

    # 左侧内容优化
    with col1:
        st.markdown(
            """
            <div style="font-size: 16px; line-height: 1.8;">
            <strong>1. 选择模型</strong><br>
            按需选择 <strong>3D</strong> 或 <strong>2D</strong> 分割模型，适配不同的医学影像场景。
            <br><br>
            <strong>2. 上传您的医学图像</strong><br>
            支持 <strong>NPZ</strong>、<strong>PNG</strong>、<strong>JPG</strong> 等多种格式，便于灵活操作。
            <br><br>
            <strong>3. 进行分割</strong><br>
            一键分割，快速获得精准的医学影像分割结果。
            </div>
            """,
            unsafe_allow_html=True,  # 允许使用HTML格式化内容
        )

    # 右侧图片优化
    with col2:
        st.image(
            "choose.png",
            caption="功能选择示意",
            use_container_width=True,
            output_format="auto",  # 自动调整格式
        )

    st.markdown("---")
    
    st.info("📢 请访问[关于](#)页面，了解更多信息。")

elif page == "3D医学图像分割":
    # 3D医学图像分割页面
    st.title("3D医学图像分割")
    st.markdown(
        """
        ### 功能介绍
        - 支持上传 3D 医学图像（例如：NPZ 格式）。
        - 提供实时分割和可视化功能。
        """
    )
    UPLOAD_DIR = "./stream_data"
    # 确保目录存在
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    # 定义进行预测的函数
    def predict_with_lite_medsam(uploaded_file_name):
        # 保存预测结果的目录
        pred_save_dir = "./stream_preds/CT_Abd"
        png_save_dir = "./stream_preds/CT_Abd_overlay"
        
        # 确保目录存在
        os.makedirs(pred_save_dir, exist_ok=True)
        os.makedirs(png_save_dir, exist_ok=True)
        
        # 执行预测命令
        command = [
            sys.executable,  # 确保使用当前 Python 环境
            #"/mount/src/litemedsam/inference_3D.py",  # 使用完整路径
            "./inference_3D.py",
            "-data_root", "stream_data", 
            "-pred_save_dir", pred_save_dir, 
            "-medsam_lite_checkpoint_path", "work_dir/medsam_lite_latest.pth", 
            "-num_workers", "4", 
            "-png_save_dir", png_save_dir, 
            "--overwrite"
        ]
        
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("命令执行成功！")
            print(result.stdout)
            
            predicted_image_path = os.path.join(png_save_dir, os.path.splitext(uploaded_file_name)[0] + ".png")
            print(predicted_image_path)
            # 检查预测结果文件是否存在
            if os.path.exists(predicted_image_path):
                return predicted_image_path
            else:
                print("预测结果图像不存在！")
                return None
        except subprocess.CalledProcessError as e:
            st.error("命令执行失败！")
            st.write(f"错误信息：{e.stderr}")
            return None

    # 上传文件
    uploaded_file = st.file_uploader("上传一个 .npz 文件", type=["npz"])

    if uploaded_file is not None:
        # 将文件保存到服务器
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"文件已成功上传并保存为: {file_path}")
        # 进行预测
        with st.spinner("正在处理，请稍候..."):
            prediction_path = predict_with_lite_medsam(uploaded_file.name)
        
        # 显示预测结果
        if prediction_path:
            st.subheader("预测结果")
            predicted_image = Image.open(prediction_path)
            st.image(predicted_image, caption="分割预测", use_container_width=True)
        else:
            st.error("预测失败！")
elif page == "2D医学图像分割":
    # 2D医学图像分割页面
    st.title("2D医学图像分割")
    st.markdown(
        """
        ### 功能介绍
        - 支持上传 2D 医学图像（例如：JPG、PNG 格式）。
        - 提供实时分割和可视化功能。
        """
    )
    uploaded_file = st.file_uploader("上传一张图片", type=["png", "jpg", "jpeg"])
    json_file_path = "./SAM-Med2D/data_demo/label2image_test.json"
    with open(json_file_path, 'w') as f:
        json.dump({}, f)

    if uploaded_file is not None:
        # 加载图片
        image = Image.open(uploaded_file)
        file_path = os.path.join("./SAM-Med2D/data_demo/images", uploaded_file.name)
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
                mask_file_name = f"./SAM-Med2D/data_demo/masks/{original_file_name}_mask_{st.session_state.idx}.png"
                mask_image.save(mask_file_name)
                # 更新 JSON 文件
                with open(json_file_path, 'w') as f:
                    json.dump({}, f)
                with open(json_file_path, 'r') as f:
                    content = json.load(f)
                    content[mask_file_name[2:]] = f"SAM-Med2D/data_demo/images/{original_file_name}.png"
                with open(json_file_path, 'w') as f_new:
                    json.dump(content, f_new, indent=4)           
                # 执行预测任务
                    command = [
                    sys.executable,
                    #"/mount/src/litemedsam/SAM-Med2D/test.py",  # 使用完整路径
                    "./SAM-Med2D/test.py",
                    "--save_pred", "True",
                    "--sam_checkpoint", "SAM-Med2D/workdir/models/sam-med2d/epoch15_sam.pth",
                ]
                with st.spinner("正在处理，请稍候..."):
                    subprocess.run(command)
                # 可视化叠加结果
                # 加载原始图像
                original_photo = f"SAM-Med2D/data_demo/images/{original_file_name}.png"
                original_image = Image.open(original_photo).convert("RGBA")  # 转为 RGBA 模式
                # 初始化透明叠加图层
                combined_image = original_image.copy()
                # 依次加载所有掩码图像并叠加
                for idx in range(len(objects)):  # 根据对象数量循环
                    generate_mask = f"./SAM-Med2D/workdir/sammed/boxes_prompt/{original_file_name}_mask_{idx}.png"
                    mask_image = Image.open(generate_mask).convert("L")  # 转为灰度图

                    # 创建掩码叠加图层
                    mask_overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))  # 创建透明图层
                    for x in range(mask_image.size[0]):
                        for y in range(mask_image.size[1]):
                            if mask_image.getpixel((x, y)) > 0:  # 如果掩码的像素值大于0，表示该位置需要叠加
                                mask_overlay.putpixel((x, y), (255, 255, 255, 128))  # 叠加半透明白色
                    # 将掩码叠加到原始图像上
                    combined_image = Image.alpha_composite(combined_image, mask_overlay)
                combined_image_path = f"./SAM-Med2D/output/{original_file_name}_combined.png"
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

elif page == "关于":
    # 关于我们页面
    st.title("关于")
    st.markdown(
        """
        **医学图像分割** 旨在为医学图像处理提供强大且高效的解决方案，以帮助医学领域更好地诊断和治疗疾病。

        ### 项目背景
        本项目作为认知科学与类脑计算的大作业，致力于将深度学习和人工智能技术与医学影像结合，实现更精确、快速的实时医学图像分割。通过对3D和2D医学图像的高效处理，帮助医生从大量影像数据中快速提取出有价值的信息，辅助诊断。

        ### 主要功能
        - **3D医学图像分割**：支持对 CT、MRI 等3D医学图像的自动分割，准确提取器官、肿瘤等重要结构。
        - **2D医学图像分割**：对医学影像的二维切片进行分割，快速分割皮肤病变、眼底图像等领域。

        ### 目标
        - 通过自动化的医学图像分析减少人工干预，提高诊断效率。
        - 实现高精度、高可靠性的医学影像分割。
        
        **感谢您的支持和关注！**
        """
    )
# 页脚
st.sidebar.markdown("---")
st.sidebar.info("© 医学图像分割平台——认知科学与类脑计算 by 董雷超")
