import streamlit as st
import whisper
import os
from openai import OpenAI

# 页面配置
st.set_page_config(page_title="AI 音频转录工具", layout="centered")
st.title("🎙️ AI 音频转录 & 润色中心")

# --- 侧边栏设置 ---
with st.sidebar:
    st.header("⚙️ 配置参数")
    api_key = st.text_input("DeepSeek API Key", type="password")
    model_size = st.selectbox("Whisper 模型大小", ["base", "small", "medium"], index=0)
    language_opt = st.radio("识别语言", ["英文 (en)", "中文 (zh)", "自动检测"])
    lang_code = "en" if "英文" in language_opt else ("zh" if "中文" in language_opt else None)

# --- 主界面 ---
uploaded_file = st.file_uploader("上传音频文件", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 1. 保存上传的文件到临时目录
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    # 输入文件名
    default_name = os.path.splitext(uploaded_file.name)[0] + "_transcribed"
    new_filename = st.text_input("保存文件名", value=default_name)
    save_path = st.text_input("保存目录 (绝对路径)", value=os.getcwd())

    if st.button("开始转录", type="primary"):
        if not api_key:
            st.error("请先在侧边栏填写 API Key！")
        else:
            with st.status("正在处理中...", expanded=True) as status:
                # 第一步：Whisper 转录
                st.write("正在加载 Whisper 模型并转录...")
                model = whisper.load_model(model_size)
                result = model.transcribe("temp_audio.mp3", language=lang_code, fp16=False)
                raw_text = result["text"]
                
                # 第二步：DeepSeek 润色
                st.write("正在调用 DeepSeek 优化文本...")
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                
                # 这里的 Prompt 会根据语言自动微调
                system_prompt = "You are a transcription editor. Paragraph the text and add punctuation. DO NOT translate." if lang_code == "en" else "你是一个速记整理员，请进行语义分段并加标点，不要翻译。"
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": raw_text}
                    ]
                )
                final_text = response.choices[0].message.content
                
                status.update(label="处理完成！", state="complete")

            # 展示结果
            st.subheader("转录结果预览")
            st.text_area("内容：", value=final_text, height=300)

            # 保存文件
            full_path = os.path.join(save_path, f"{new_filename}.txt")
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            st.success(f"文件已成功保存至: {full_path}")