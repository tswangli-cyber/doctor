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

```bash
# 克隆仓库
git clone https://github.com/tswangli-cyber/doctor.git

# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
```

## 许可证

MIT License
