python -m pytest -q tests
python run_e2e_test.py
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "clientInfo": {"name": "cli-client", "version": "1.0.0"}}}' | python3 mcp_from_scratch.py 
echo '{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}' | python3 mcp_from_scratch.py 
echo '{"jsonrpc": "2.0", "method": "tools/call", "id": 2, "params": {"name": "read_document_with_mcp", "arguments": {"layer": "frontend"}}}' | python3 mcp_from_scratch.py