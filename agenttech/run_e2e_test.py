#!/usr/bin/env python3
"""
E2E 测试运行脚本
从 aiworkshop 目录下运行端到端测试
"""

import sys
import asyncio
from pathlib import Path
import dotenv
dotenv.load_dotenv()
# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from aiworkshop.tests.integration_tests.e2e_test import E2ETestEvaluator

async def main():
    """主函数"""
    print("🚀 开始运行 E2E 测试...")
    print(f"📁 当前工作目录: {Path.cwd()}")
    
    evaluator = E2ETestEvaluator()
    results = await evaluator.run_complete_evaluation()
    
    if results:
        print("✅ E2E 测试完成！")
        print(f"📊 报告路径: {results.get('report_path', 'N/A')}")
    else:
        print("❌ E2E 测试失败！")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
