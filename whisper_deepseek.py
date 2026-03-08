import os
import whisper
from openai import OpenAI

# ==========================
# 1️⃣ 设置 DeepSeek API
# ==========================
# 请去 https://platform.deepseek.com/ 创建 API Key
# 账户里充值 2-5 元人民币就能用很久很久
DEEPSEEK_API_KEY = "API_KEY_HERE"  # <-- 替换为你的 DeepSeek API Key

client = OpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com"
)

# ==========================
# 2️⃣ 设置音频路径
# ==========================
AUDIO_FILE = r"C:\Users\yanzh\Documents\YanZ\2_Guidepoint\Putuo.mp3"

# ==========================
# 3️⃣ 加载 Whisper 模型 & 转写
# ==========================
# 使用 base 模型，添加 initial_prompt 引导识别
model = whisper.load_model("base")
print("正在本地转写音频（这可能需要几分钟）...")

# fp16=False 是为了避免 CPU 运行时的警告
result = model.transcribe(
    AUDIO_FILE, 
    language="en", 
    fp16=False,
    initial_prompt="This is a professional interview. Please include punctuation and capitalize proper nouns and notice terms。"
)
text = result["text"]
print("✅ 原始转写完成！")

# ==========================
# 4️⃣ 使用 DeepSeek 自动分段 & 润色
# ==========================
print("正在调用 DeepSeek 进行语义分段...")

# 注意：如果音频超过 1 小时，文本可能过长，这里建议分段发送（目前先按全量处理）
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a professional transcript editor. Your task is to format the original text into paragraphs, correct minor transcription errors, and add proper punctuation. DO NOT translate the text. Keep it in its original language."},
            {"role": "user", "content": f"Please format and proofread the following transcript:\n\n{text}"}
        ],
        stream=False
    )

    segmented_text = response.choices[0].message.content
    print("✅ DeepSeek 处理完成！")

except Exception as e:
    print(f"❌ AI 处理出错: {e}")
    segmented_text = text  # 如果 AI 出错，回退到原始文本

# ==========================
# 5️⃣ 保存文本
# ==========================
output_txt = os.path.splitext(AUDIO_FILE)[0] + "_final.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(segmented_text)

print(f"🚀 任务完成！结果已保存至: {output_txt}")