import os
import requests
import logging

logger = logging.getLogger("RepoRadar.GLMAPI")

class GLMFlashAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.example.com/v1"  # 替换为实际的 API 基础 URL

    def query(self, prompt: str) -> str:
        """使用 glm-4-flash API 进行查询"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "glm-4-flash",
            "messages": [
                {"role": "system", "content": "你是一个智能助手，根据提供的内容准确回答问题。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error(f"API调用失败：{str(e)}")
            return "API调用失败"
