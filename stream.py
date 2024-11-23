import subprocess
import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
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

st.title("LiteMedSAM 图像分割预测")

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
