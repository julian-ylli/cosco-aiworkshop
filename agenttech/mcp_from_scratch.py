import asyncio
import json
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional
import os
import inspect # Import inspect for dynamic tool registration

current_file_path = os.path.abspath(__file__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPHandler:
    """MCP协议处理器"""
    
    def __init__(self):
        """初始化MCP处理器并注册工具"""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_tool(self.read_document_with_mcp)

    def _register_tool(self, method):
        """动态注册工具"""
        tool_name = method.__name__
        description = inspect.getdoc(method) or ""
        
        # Simple input schema generation for demonstration
        # In a real scenario, you might use type hints or a more sophisticated schema generator
        signature = inspect.signature(method)
        properties = {}
        for name, param in signature.parameters.items():
            if name == 'self':
                continue
            properties[name] = {"type": "string"} # Assuming all params are strings for simplicity
        
        self.tools[tool_name] = {
            "name": tool_name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties
            }
        }

    def read_document_with_mcp(self, layer: str) -> str:
        """Finds the tech document based on the layer.

        Args:
            layer: the layer of tech document to find only include "frontend", "backend"
        """
        try:
            if layer == "frontend":
                with open(os.path.join(os.path.dirname(current_file_path), "tech_doc/frontend.md"), "r", encoding="utf-8") as file:
                    return file.read()  # read file from frontend.md
            elif layer == "backend":
                with open(os.path.join(os.path.dirname(current_file_path), "tech_doc/backend.md"), "r", encoding="utf-8") as file:
                    return file.read()  # read file from backend.md
            else:
                raise ValueError("Unsupported layer. Only 'frontend' and 'backend' are allowed.")
        except FileNotFoundError:
            return f"文档文件 {layer}.md 不存在"
        except Exception as e:
            return f"读取文档时出错: {str(e)}"
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP请求"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "initialize":
                return self._create_response(request_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "aiworkshop-mcp-server", "version": "1.0.0"}
                })
            
            elif method == "tools/list":
                return self._create_response(request_id, {"tools": list(self.tools.values())})
            
            elif method == "notifications/initialized":
                return self._create_response(request_id, {})
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name not in self.tools:
                    return self._create_error_response(request_id, -32601, f"Method not found: {tool_name}")
                
                tool_method = getattr(self, tool_name, None)
                if not tool_method:
                    return self._create_error_response(request_id, -32601, f"Unknown tool: {tool_name}")
                
                result = tool_method(**arguments)
                return self._create_response(request_id, {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
                })
            
            else:
                return self._create_error_response(request_id, -32601, f"Method not found: {method}")
                
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            logger.error(traceback.format_exc())
            return self._create_error_response(request_id, -32603, f"Internal error: {str(e)}")

    def _create_response(self, request_id: Optional[Any], result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _create_error_response(self, request_id: Optional[Any], code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
async def main():
    """主函数"""
    handler = MCPHandler()
    
    # 读取标准输入
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # 解析JSON请求
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                continue
            
            # 处理请求
            response = await handler.handle_request(request)
            
            # 输出响应
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            break

if __name__ == "__main__":
    asyncio.run(main())
