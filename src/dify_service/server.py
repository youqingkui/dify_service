import os
from typing import Any, Sequence
import logging

import httpx
import asyncio
from dotenv import load_dotenv
from mcp.server import Server, NotificationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    EmptyResult
)
from pydantic import AnyUrl

# 设置日志
logger = logging.getLogger("dify-server")
logger.setLevel(logging.INFO)

# 加载环境变量
load_dotenv()

# API配置
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "https://api.dify.ai/v1")

if not DIFY_API_KEY:
    raise ValueError("DIFY_API_KEY environment variable required")

app = Server("dify_service")

@app.set_logging_level()
async def set_logging_level(level: LoggingLevel) -> EmptyResult:
    """设置日志级别"""
    logger.setLevel(level.upper())
    await app.request_context.session.send_log_message(
        level="debug",
        data=f"Log level set to {level}",
        logger="dify-server"
    )
    return EmptyResult()

async def query_knowledge(query: str) -> dict[str, Any]:
    """查询Dify知识库"""
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    logger.debug("Creating HTTP client...")
    try:
        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            logger.debug(f"Sending query to Dify API: {query}")
            
            request_data = {
                "inputs": {},
                "query": query,
                "response_mode": "blocking",
                "user": "mcp-user"
            }
            logger.debug(f"Request data: {request_data}")
            
            response = await client.post(
                f"{DIFY_API_BASE}/chat-messages",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response data: {result}")
            return result
            
    except httpx.TimeoutException:
        logger.error("Request timed out")
        raise RuntimeError("Request timed out")
    except httpx.HTTPError as e:
        logger.error(f"HTTP error: {str(e)}")
        raise RuntimeError(f"API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError(f"Unexpected error: {str(e)}")

@app.list_resources()
async def list_resources() -> list[Resource]:
    """列出知识库查询功能"""
    return [
        Resource(
            uri=AnyUrl("dify://knowledge"),
            name="Dify Knowledge Base",
            description="Query Dify knowledge base",
            mimeType="text/plain",
        )
    ]

@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """读取知识库信息"""
    if uri.scheme != "dify":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")
    return "Use the query-knowledge tool to search the knowledge base."

@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="query-knowledge",
            description="查询Dify知识库",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )
    ]

@app.call_tool()
async def call_tool(
    name: str, arguments: dict | None
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """处理工具调用请求"""
    logger.info(f"Tool call received: {name}")
    logger.debug(f"Tool arguments: {arguments}")

    if name != "query-knowledge":
        raise ValueError(f"Unknown tool: {name}")

    if not arguments or "query" not in arguments:
        raise ValueError("Missing query argument")

    # 安全地获取进度令牌
    progress_token = None
    if hasattr(app.request_context, 'meta') and app.request_context.meta:
        progress_token = getattr(app.request_context.meta, 'progressToken', None)

    if progress_token:
        # 发送开始进度通知
        await app.request_context.session.send_progress_notification(
            progress_token=progress_token,
            progress=0,
            total=1
        )

    try:
        # 调用Dify API并处理结果
        result = await query_knowledge(arguments["query"])
        
        if progress_token:
            # 发送完成进度通知
            await app.request_context.session.send_progress_notification(
                progress_token=progress_token,
                progress=1,
                total=1
            )
        
        return [
            TextContent(
                type="text",
                text=result.get("answer", "No answer found"),
            )
        ]
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        if progress_token:
            # 发送错误进度通知
            await app.request_context.session.send_progress_notification(
                progress_token=progress_token,
                progress=1,
                total=1,
                error=str(e)
            )
        raise

async def main():
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Dify MCP server...")
    try:
        logger.debug("Initializing stdio server...")
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server transport initialized")
            logger.debug("Starting server run...")
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
            logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())