import os
from typing import Any, Sequence
import logging
import sys
import json

import httpx
import asyncio
from dotenv import load_dotenv
from mcp.shared.exceptions import McpError
from mcp.server import Server
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
    except asyncio.CancelledError:
        logger.error("Request cancelled by client", exc_info=True)
        raise McpError(INTERNAL_ERROR, "Request cancelled")
    except Exception as e:
        logger.error(f"Unexpected error in query_knowledge: {str(e)}", exc_info=True)
        raise McpError(INTERNAL_ERROR, f"Unexpected error: {str(e)}")

@app.list_resources()
async def list_resources() -> list[Resource]:
    """列出知识库查询功能"""
    return [
        Resource(
            uri=AnyUrl("dify://test/knowledge"),
            name="Dify Knowledge Base",
            description="Query Dify knowledge base. You can specify test query in URI like: dify://<query>/knowledge",
            mimeType="text/plain",
        )
    ]

@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """读取知识库信息"""
    logger.debug(f"Reading resource: {uri}")
    
    if str(uri).startswith("dify://") and str(uri).endswith("/knowledge"):
        try:
            # 从 URI 解析查询内容
            uri_parts = str(uri).split("/")
            test_query = uri_parts[-2] if len(uri_parts) > 2 else "test connection"
            logger.debug(f"Using test query from URI: {test_query}")
            
            # 使用 query_knowledge 测试 API
            result = await query_knowledge(test_query)
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error reading resource: {str(e)}", exc_info=True)
            raise McpError(INTERNAL_ERROR, f"Failed to read resource: {str(e)}")
    else:
        logger.warning(f"Invalid resource URI: {uri}")
        raise McpError(INVALID_PARAMS, f"Invalid resource URI: {uri}")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="query-dify-knowledge",
            description="Query Dify knowledge base",
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
    
    if name != "query-dify-knowledge":
        logger.warning(f"Unknown tool requested: {name}")
        raise McpError(INVALID_PARAMS, f"Unknown tool: {name}")

    if not arguments or "query" not in arguments:
        logger.warning("Missing query in arguments")
        raise McpError(INVALID_PARAMS, "Missing query argument")

    try:
        logger.info(f"Starting knowledge query with: {arguments['query']}")
        result = await query_knowledge(arguments["query"])
        logger.debug(f"Query completed successfully, result: {result}")
        response = [
            TextContent(
                type="text",
                text=result.get("answer", "No answer found"),
            )
        ]
        logger.info("Tool call completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error during query execution: {str(e)}", exc_info=True)
        raise McpError(INTERNAL_ERROR, str(e))

async def main():
    from mcp.server.stdio import stdio_server    
    logger.debug("Initializing stdio server...")
    options = app.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            options,
            raise_exceptions=True  # 抛出异常
        )
        logger.info("Server started successfully")

if __name__ == "__main__":
    asyncio.run(main())