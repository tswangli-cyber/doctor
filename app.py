"""
Doctor - 医疗问诊与临床决策AI系统
基于 HuatuoGPT 的智能医疗问答服务
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ============ 配置 ============
MODEL_NAME = os.getenv("MODEL_NAME", "FreedomIntelligence/HuatuoGPT2-7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# ============ FastAPI 应用 ============
app = FastAPI(
    title="Doctor API",
    description="医疗问诊与临床决策AI系统 - 基于HuatuoGPT",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 数据模型 ============
class ConsultationRequest(BaseModel):
    """问诊请求"""
    symptoms: str  # 患者症状描述
    history: Optional[str] = ""  # 既往病史
    age: Optional[int] = None  # 年龄
    gender: Optional[str] = None  # 性别
    conversation_history: Optional[List[dict]] = []  # 对话历史


class ConsultationResponse(BaseModel):
    """问诊响应"""
    response: str  # AI回复
    suggested_questions: List[str]  # 建议追问的问题
    possible_conditions: List[str]  # 可能的诊断
    urgency_level: str  # 紧急程度: low, medium, high


# ============ 模型加载 ============
class MedicalModel:
    """医疗大模型封装"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loaded = False
    
    def load(self):
        """加载模型"""
        if self.loaded:
            return
        
        print(f"正在加载模型: {MODEL_NAME}")
        print(f"设备: {DEVICE}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True
            )
            
            if DEVICE == "cpu":
                self.model = self.model.to(DEVICE)
            
            self.model.eval()
            self.loaded = True
            print("模型加载完成")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用演示模式")
            self.loaded = False
    
    def generate(self, prompt: str, max_length: int = MAX_LENGTH) -> str:
        """生成回复"""
        if not self.loaded:
            return self._demo_response(prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def _demo_response(self, prompt: str) -> str:
        """演示模式响应"""
        if "症状" in prompt or "不舒服" in prompt:
            return """根据您描述的症状，我需要了解更多信息：

1. 这种症状持续多长时间了？
2. 疼痛是持续性的还是间歇性的？
3. 有没有伴随发热、恶心等其他症状？
4. 最近有没有吃过特殊的食物或药物？

请注意：以上只是初步问诊建议，不能替代专业医生的诊断。如症状严重，请及时就医。"""
        
        return "您好，我是医疗问诊助手。请描述您的症状，我会尽力帮助您分析。"


# 全局模型实例
medical_model = MedicalModel()


# ============ API 端点 ============
@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    medical_model.load()


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": medical_model.loaded,
        "model_name": MODEL_NAME,
        "device": DEVICE
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}


@app.post("/consult", response_model=ConsultationResponse)
async def consult(request: ConsultationRequest):
    """医疗问诊接口"""
    
    # 构建提示词
    prompt = f"""你是一位专业的医疗问诊助手。请根据以下信息进行分析：

患者信息：
- 年龄：{request.age if request.age else '未知'}
- 性别：{request.gender if request.gender else '未知'}
- 既往病史：{request.history if request.history else '无'}

主诉症状：
{request.symptoms}

请提供：
1. 针对性的追问
2. 可能的诊断方向
3. 建议

回复："""
    
    try:
        response = medical_model.generate(prompt)
        
        # 解析响应（简化版，实际应用需要更复杂的解析逻辑）
        return ConsultationResponse(
            response=response,
            suggested_questions=[
                "这种症状持续多长时间了？",
                "有没有加重或缓解的因素？",
                "有没有伴随其他症状？"
            ],
            possible_conditions=["需要更多信息进行判断"],
            urgency_level="medium"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回复时出错: {str(e)}")


# ============ Gradio 界面 ============
def create_gradio_interface():
    """创建 Gradio 聊天界面"""
    
    def chat(message, history):
        """聊天函数"""
        prompt = f"患者：{message}\n医生助手："
        response = medical_model.generate(prompt)
        return response
    
    with gr.Blocks(title="Doctor - 医疗问诊助手") as demo:
        gr.Markdown("""
        # 🩺 Doctor - 医疗问诊助手
        
        基于 HuatuoGPT 的智能医疗问答系统
        
        > ⚠️ **免责声明**：本系统仅供辅助参考，不能替代专业医生的诊断。如有严重症状，请及时就医。
        """)
        
        gr.ChatInterface(
            chat,
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(placeholder="请描述您的症状...", container=False, scale=7),
            additional_inputs=[],
        )
        
        gr.Markdown("""
        ---
        ### 使用提示
        
        - 详细描述您的症状
        - 提供年龄、性别等基本信息
        - 说明症状持续的时间
        - 提及是否有既往病史
        """)
    
    return demo


# ============ 主函数 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Doctor 医疗问诊系统")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    parser.add_argument("--webui", action="store_true", help="启动 Gradio 界面")
    parser.add_argument("--demo", action="store_true", help="演示模式（不加载模型）")
    args = parser.parse_args()
    
    if args.demo:
        print("演示模式：不会加载实际模型")
    
    if args.webui:
        print(f"启动 Gradio 界面: http://{args.host}:{args.port}")
        demo = create_gradio_interface()
        demo.launch(server_name=args.host, server_port=args.port)
    else:
        import uvicorn
        print(f"启动 FastAPI 服务: http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
