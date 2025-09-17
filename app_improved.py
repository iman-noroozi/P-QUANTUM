import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "pqn-ai/pqn-ai-v1"

# برای CPU پایدار
torch.set_num_threads(max(1, torch.get_num_threads() - 0))  # اختیاراً محدود کن

print("=== Loading پی یار (Pey Yar) - PQN.AI ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float32,      # روی CPU امن‌تره
    device_map="cpu",
    low_cpu_mem_usage=True
).eval()

SYSTEM_PROMPT_FA = (
    "تو «پی‌یار» هستی: دستیار کوانتومی فارسی‌دان، مودب، دقیق و کوتاه‌نویس. "
    "اگر سؤال مبهم بود، یک توضیح کوتاه بپرس. از ادعاهای تاییدنشده پرهیز کن."
)
SYSTEM_PROMPT_EN = (
    "You are 'PiYar': a concise, helpful, Persian-first quantum assistant. "
    "Ask a brief clarifying question if the user prompt is ambiguous. Avoid unverifiable claims."
)

def build_chat_prompt(message, history):
    """تبدیل تاریخچه به پرامپت خطی پایدار برای مدل‌های بدون chat_template"""
    lines = [f"[SYSTEM_FA] {SYSTEM_PROMPT_FA}", f"[SYSTEM_EN] {SYSTEM_PROMPT_EN}"]
    for user, assistant in history or []:
        if user:
            lines.append(f"[USER] {user}".strip())
        if assistant:
            lines.append(f"[ASSISTANT] {assistant}".strip())
    lines.append(f"[USER] {message}".strip())
    lines.append("[ASSISTANT]")
    return "\n".join(lines)

def generate(message, history, temperature, top_p, top_k, max_new_tokens, repetition_penalty):
    try:
        # ساخت پرامپت
        prompt = build_chat_prompt(message, history)

        # توکنایز با محدودیت طول امن
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                max_new_tokens=int(max_new_tokens),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # بخش جدید را جدا کن
        gen_ids = output_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # تمیزکاری خروجی
        text = text.strip()
        if not text:
            text = "خروجی نامعتبر بود؛ لطفاً دوباره تلاش کن یا جمله را کمی بازنویسی کن."

        return text

    except Exception as e:
        print(f"Error in generation: {e}")
        return f"خطا در تولید پاسخ: {str(e)}"

# Create premium chat interface with Pey Yar logo
demo = gr.ChatInterface(
    fn=generate,
    title="🟣 پی یار (Pey Yar) - PQN.AI",
    description="تجربه نسخه نمایشی: پاسخ کوتاه، دقیق و فارسی‌اول. (CPU Demo)",
    chatbot=gr.Chatbot(
        avatar_images=["https://huggingface.co/spaces/pqn-ai/pqn-ai-demo/resolve/main/pey_yar_logo.png", None],
        height=400
    ),
    additional_inputs=[
        gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.3, 1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Slider(0, 100, value=40, step=5, label="Top-k"),
        gr.Slider(16, 1024, value=256, step=16, label="Max new tokens"),
        gr.Slider(1.0, 1.5, value=1.08, step=0.01, label="Repetition penalty"),
    ],
    examples=[
        "سلام! یک داستان خیلی کوتاه علمی‌-تخیلی بنویس.",
        "یک شعر سه‌سطری درباره «امید کوانتومی» بگو.",
        "در یک پاراگراف، پی یار را معرفی کن.",
    ],
)

if __name__ == "__main__":
    demo.launch()
