import whisper
import os

# 配置参数
AUDIO_PATH = "./audio/dubbing.wav"  # 任务二导出的音频路径
MODEL_NAME = "base"  # 可选tiny/base/small/medium/large
OUTPUT_PATH = "./asr_result.txt"

def main():
    # 1. 加载模型（第一次运行会自动下载，约1G左右）
    print(f"正在加载Whisper {MODEL_NAME}模型...")
    model = whisper.load_model(MODEL_NAME)
    
    # 2. 检查音频文件
    if not os.path.exists(AUDIO_PATH):
        print(f"错误：音频文件不存在，请检查路径 {AUDIO_PATH}")
        return
    
    # 3. 执行语音识别
    print(f"正在识别音频 {AUDIO_PATH}...")
    result = model.transcribe(AUDIO_PATH, language="zh")
    
    # 4. 保存识别结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    # 5. 打印结果
    print("\n识别结果：")
    print(result["text"])
    print(f"\n结果已保存到 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()