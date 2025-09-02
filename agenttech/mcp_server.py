from mcp.server.fastmcp import FastMCP
import asyncio
import os
current_file_path = os.path.abspath(__file__)
# Create an MCP server
mcp = FastMCP("Demo")

# Add an addition tool
@mcp.tool()
def read_document_with_mcp(layer: str) -> str:
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

def main():
    """Entry point for the direct execution server."""
    mcp.run()


if __name__ == "__main__":
    main()