# Doctor - 医疗问诊与临床决策AI系统

基于 HuatuoGPT 的智能医疗问答系统

## 功能特点

- 🩺 **智能问诊**：多轮对话采集病史，模拟真实医患交流
- 📊 **临床决策**：辅助诊断建议，提供鉴别诊断思路
- 💊 **用药指导**：药物相互作用检查，用药注意事项提醒
- 📋 **病历生成**：自动生成结构化问诊记录

## 技术架构

| 组件 | 技术选型 |
|------|----------|
| 基础模型 | HuatuoGPT-II |
| 部署框架 | FastAPI + Docker |
| 前端界面 | Gradio |
| 向量数据库 | Milvus (医学知识库) |

## 快速开始

### 方式一：直接运行

```bash
# 克隆仓库
git clone https://github.com/tswangli-cyber/doctor.git
cd doctor

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp .env.example .env

# 启动 API 服务
python app.py

# 或启动 Web 界面
python app.py --webui
```

### 方式二：Docker 部署

```bash
# 构建镜像
docker build -t doctor:latest .

# 运行容器
docker run -d \
  --name doctor \
  -p 8000:8000 \
  -p 7860:7860 \
  --gpus all \
  doctor:latest

# 演示模式（无需 GPU）
docker run -d -p 8000:8000 doctor:latest python app.py --demo
```

## API 使用

### 健康检查

```bash
curl http://localhost:8000/health
```

### 问诊接口

```bash
curl -X POST http://localhost:8000/consult \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "头痛、发热两天",
    "age": 35,
    "gender": "男",
    "history": "无特殊病史"
  }'
```

### 响应示例

```json
{
  "response": "根据您描述的症状...",
  "suggested_questions": [
    "体温是多少度？",
    "有没有咳嗽、流鼻涕？",
    "头痛是持续性的还是阵发性的？"
  ],
  "possible_conditions": ["感冒", "流感", "偏头痛"],
  "urgency_level": "medium"
}
```

## 项目结构

```
doctor/
├── app.py              # 主应用
├── requirements.txt    # Python 依赖
├── Dockerfile           # Docker 配置
├── .env.example        # 环境变量模板
└── README.md           # 本文件
```

## 模型说明

本系统使用 **HuatuoGPT-II**，由华中科技大学与香港中文大学联合开发的中文医疗大模型。

- 模型大小：7B 参数
- 训练数据：中文医学文献、问诊对话
- 能力：问诊对话、诊断建议、用药指导

> ⚠️ **免责声明**：本系统仅供辅助参考，不能替代专业医生的诊断。如有严重症状，请及时就医。

## 许可证

MIT License
