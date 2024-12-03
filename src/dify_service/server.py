import os
from typing import Any, Sequence
import logging
import sys

import httpx
import asyncio
from dotenv import load_dotenv
from mcp.shared.exceptions import McpError
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    EmptyResult,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
import mcp.types as types
from pydantic import AnyUrl

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s',
    handlers=[
        # 控制台处理器
        logging.StreamHandler(sys.stderr),
        # 文件处理器
        logging.FileHandler('dify_service.log')
    ]
)
logger = logging.getLogger("dify-server")

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
    logger.info(f"Starting Dify API query: {query}")
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    logger.debug("Creating HTTP client...")
    try:
        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            logger.debug("HTTP client created successfully")
            logger.debug(f"Preparing request to Dify API: {query}")
            
            request_data = {
                "inputs": {},
                "query": query,
                "response_mode": "blocking",
                "user": "mcp-user"
            }
            logger.debug(f"Request data prepared: {request_data}")
            
            logger.info(f"Sending request to {DIFY_API_BASE}/chat-messages")
            response = await client.post(
                f"{DIFY_API_BASE}/chat-messages",
                json=request_data
            )
            logger.debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response parsed successfully: {result}")
            return result
            
    except httpx.TimeoutException:
        logger.error("Request timed out while calling Dify API", exc_info=True)
        raise McpError(INTERNAL_ERROR, "Request timed out")
    except Exception as e:
        logger.error(f"Unexpected error in query_knowledge: {str(e)}", exc_info=True)
        raise McpError(INTERNAL_ERROR, f"Unexpected error: {str(e)}")

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
        raise McpError(INVALID_PARAMS, f"Unsupported URI scheme: {uri.scheme}")
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
    logger.info(f"Tool call started - name: {name}, arguments: {arguments}")
    try:
        if name != "query-knowledge":
            logger.warning(f"Unknown tool requested: {name}")
            raise McpError(INVALID_PARAMS, f"Unknown tool: {name}")

        if not arguments or "query" not in arguments:
            logger.warning("Missing query in arguments")
            raise McpError(INVALID_PARAMS, "Missing query argument")

        # 获取进度令牌
        progress_token = None
        if hasattr(app.request_context, 'meta'):
            progress_token = getattr(app.request_context.meta, 'progressToken', None)
            logger.debug(f"Progress token obtained: {progress_token}")

        try:
            if progress_token:
                logger.debug(f"Sending initial progress notification with token: {progress_token}")
                await app.request_context.session.send_progress_notification(
                    progress_token=progress_token,
                    progress=0,
                    total=1
                )
                logger.debug("Initial progress notification sent successfully")

            logger.info(f"Starting knowledge query with: {arguments['query']}")
            result = await query_knowledge(arguments["query"])
            logger.debug(f"Query completed successfully, result: {result}")

            if progress_token:
                logger.debug("Sending completion progress notification")
                await app.request_context.session.send_progress_notification(
                    progress_token=progress_token,
                    progress=1,
                    total=1
                )
                logger.debug("Completion progress notification sent successfully")

            response = [
                TextContent(
                    type="text",
                    text=result.get("answer", "No answer found"),
                )
            ]
            logger.info("Tool call completed successfully")
            return response

        except asyncio.CancelledError:
            logger.error("Request cancelled by client")
            if progress_token:
                try:
                    logger.debug("Sending cancellation progress notification")
                    await app.request_context.session.send_progress_notification(
                        progress_token=progress_token,
                        progress=1,
                        total=1,
                        error="Request cancelled"
                    )
                    logger.debug("Cancellation progress notification sent")
                except anyio.ClosedResourceError:
                    logger.warning("Could not send cancellation notification - resource closed")
            raise McpError(INTERNAL_ERROR, "Request cancelled")
            
        except Exception as e:
            logger.error(f"Error during query execution: {str(e)}", exc_info=True)
            if progress_token:
                try:
                    logger.debug("Sending error progress notification")
                    await app.request_context.session.send_progress_notification(
                        progress_token=progress_token,
                        progress=1,
                        total=1,
                        error=str(e)
                    )
                    logger.debug("Error progress notification sent")
                except anyio.ClosedResourceError:
                    logger.warning("Could not send error notification - resource closed")
            if isinstance(e, McpError):
                raise
            raise McpError(INTERNAL_ERROR, str(e))

    except Exception as e:
        logger.error(f"Outer exception handler caught: {str(e)}", exc_info=True)
        if isinstance(e, McpError):
            raise
        raise McpError(INTERNAL_ERROR, str(e))

async def main():
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Dify MCP server...")
    try:
        logger.debug("Initializing stdio server...")
        options = app.create_initialization_options()
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server transport initialized")
            logger.debug("Starting server run...")
            try:
                await app.run(
                    read_stream,
                    write_stream,
                    options,
                    raise_exceptions=True
                )
                logger.info("Server started successfully")
            except anyio.ClosedResourceError as e:
                logger.warning(f"Resource closed: {str(e)}")
                return  # 正常退出
            except Exception as e:
                logger.error(f"Error during server run: {str(e)}", exc_info=True)
                raise McpError(INTERNAL_ERROR, str(e))
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise McpError(INTERNAL_ERROR, str(e))

if __name__ == "__main__":
    asyncio.run(main())