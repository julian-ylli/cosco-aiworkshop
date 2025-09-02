import dotenv
import pandas as pd
import numpy as np
import asyncio
import nest_asyncio
import os
from datetime import datetime
import json
from pathlib import Path


# 加载环境变量
dotenv.load_dotenv()

# 导入必要的模块
from agent_tech_QA_MCP import graph
from langchain_core.messages import HumanMessage, ToolMessage
from ragas import EvaluationDataset
from ragas.metrics import ResponseRelevancy

from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
import uuid

class E2ETestEvaluator:
    def __init__(self):
        """初始化评估器"""
        # 使用 Google Gemini 作为评估 LLM，确保 API key 已设置
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("⚠️  GOOGLE_API_KEY 环境变量未设置，RAGAS 评估可能失败")

        self.evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=google_api_key))
        self.embedding_llm = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=google_api_key)
        self.metrics = [
              ResponseRelevancy(llm=self.evaluator_llm, embeddings=self.embedding_llm),
          ]
        
    async def load_test_data(self, csv_path="tests/data/agent_tech_QA_MCP_data.csv"):
        """加载测试数据"""
        try:
            synth_data = pd.read_csv(csv_path)
            questions = synth_data["questions"].tolist()
            expected_responses = synth_data["responses"].tolist() if "responses" in synth_data.columns else [None]*len(questions)
            expected_trajectories = synth_data["tool_trajectory"].tolist() if "tool_trajectory" in synth_data.columns else [None]*len(questions)
            return questions, expected_responses, expected_trajectories
        except FileNotFoundError:
            print(f"❌ 测试数据文件不存在: {csv_path}")
            return [], []
        except KeyError as e:
            print(f"❌ 测试数据文件格式错误，缺少列: {e}")
            return [], [], []

    async def run_e2e_tests(self, questions, expected_trajectories=None, expected_responses=None):
        """运行端到端测试"""
        outputs = {
            "questions": questions,
            "expected_responses": expected_responses or [],
            "expected_trajectories": expected_trajectories or [],
            "responses": [],
            "tool_trajectory": [],
            "full_trajectory": [],
            "execution_time": []
        }
        
        print(f"🚀 开始运行 {len(questions)} 个端到端测试...")
        
        for i, q in enumerate(questions):
            print(f"📝 处理问题 {i+1}/{len(questions)}: {q[:50]}...")
            
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            input_message = HumanMessage(content=q)
            
            # 记录执行时间
            start_time = datetime.now()
            
            try:
                output = await graph.ainvoke({"messages": [input_message]}, config)
                
                # 提取响应和轨迹
                final_response = output["messages"][-1].content
                tool_trajectory = "[TOOLSEP]".join([
                    m.name for m in output["messages"] 
                    if isinstance(m, ToolMessage)
                ])
                
                # 记录完整轨迹
                full_trajectory = []
                for msg in output["messages"]:
                    if isinstance(msg, ToolMessage):
                        full_trajectory.append(f"TOOL:{msg.name}")
                    else:
                        full_trajectory.append(f"MSG:{type(msg).__name__}")
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                outputs["responses"].append(final_response)
                outputs["tool_trajectory"].append(tool_trajectory)
                outputs["full_trajectory"].append(" -> ".join(full_trajectory))
                outputs["execution_time"].append(execution_time)
                
                print(f"✅ 完成 - 响应长度: {len(final_response)}, 工具调用: {len(tool_trajectory.split('[TOOLSEP]')) if tool_trajectory else 0}")
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                outputs["responses"].append(f"ERROR: {str(e)}")
                outputs["tool_trajectory"].append("")
                outputs["full_trajectory"].append("ERROR")
                outputs["execution_time"].append(0)
        
        return outputs
    
    async def evaluate_with_ragas(self, outputs):
        """使用 RAGAS 评估答案质量，确保对比期望输出和实际输出"""
        print("🔍 开始 RAGAS 评估...")
        results = {}
        eval_data = []
        expected_responses = outputs.get("expected_responses", [])
        for i, (question, response) in enumerate(zip(outputs["questions"], outputs["responses"])):
            if not response.startswith("ERROR"):
                ground_truth = expected_responses[i] if expected_responses and i < len(expected_responses) else ""
                eval_data.append({
                    "question": str(question),
                    "answer": str(response),
                    "ground_truth": ground_truth
                })
        if not eval_data:
            print("❌ 没有有效数据用于评估")
            return results
        try:
            from ragas import SingleTurnSample
            samples = []
            for item in eval_data:
                try:
                    sample = SingleTurnSample(
                        user_input=item["question"],
                        response=item["answer"],
                        reference=item["ground_truth"]
                    )
                    samples.append(sample)
                except Exception as e:
                    print(f"❌ 创建样本失败: {e}")
                    print(f"   问题: {item['question'][:50]}...")
                    continue
            if not samples:
                print("❌ 没有有效的样本用于评估")
                return results
            dataset = EvaluationDataset(samples)
            from ragas import evaluate
            print(f"📊 开始评估 {len(samples)} 个样本...")
            evaluation_results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                # llm=self.evaluator_llm,
                show_progress=True
            )
            for metric in self.metrics:
                try:
                    score = evaluation_results[metric.name]
                    # 如果是list，取均值用于打印和HTML展示
                    if isinstance(score, list):
                        score_value = float(np.mean(score)) if len(score) > 0 else 0.0
                        results[metric.name] = score  # 保留原始list用于后续HTML每条用例展示
                    else:
                        score_value = float(score)
                        results[metric.name] = score
                    print(f"   {metric.name}: {score_value:.4f}")
                except Exception as e:
                    print(f"❌ 提取 {metric.name} 分数失败: {e}")
                    results[metric.name] = None
        except Exception as e:
            print(f"❌ RAGAS 评估失败: {e}")
            results["error"] = str(e)
        return results
    
    def analyze_trajectory(self, outputs):
        """分析工具调用轨迹"""
        print("🛤️  分析工具调用轨迹...")
        
        trajectory_analysis = {
            "total_questions": len(outputs["questions"]),
            "successful_runs": len([r for r in outputs["responses"] if not r.startswith("ERROR")]),
            "tool_usage_stats": {},
            "trajectory_patterns": {},
            "execution_time_stats": {}
        }
        
        # 分析工具使用情况
        all_tools = []
        for trajectory in outputs["tool_trajectory"]:
            if trajectory:
                tools = trajectory.split("[TOOLSEP]")
                all_tools.extend(tools)
        
        from collections import Counter
        tool_counts = Counter(all_tools)
        trajectory_analysis["tool_usage_stats"] = dict(tool_counts)
        
        # 分析执行时间
        valid_times = [t for t in outputs["execution_time"] if t > 0]
        if valid_times:
            trajectory_analysis["execution_time_stats"] = {
                "avg_time": sum(valid_times) / len(valid_times),
                "min_time": min(valid_times),
                "max_time": max(valid_times),
                "total_time": sum(valid_times)
            }
        
        # 分析轨迹模式
        trajectory_patterns = Counter(outputs["full_trajectory"])
        trajectory_analysis["trajectory_patterns"] = dict(trajectory_patterns)
        
        return trajectory_analysis
    
    def analyze_trajectory_accuracy(self, outputs):
        """分析轨迹准确率"""
        print("🎯 分析轨迹准确率...")
        
        if not outputs.get("expected_trajectories"):
            print("⚠️  没有预期轨迹数据，跳过轨迹准确率分析")
            return {
                "trajectory_accuracy": 0.0,
                "total_comparisons": 0,
                "correct_trajectories": 0,
                "trajectory_details": []
            }
        
        trajectory_accuracy_analysis = {
            "trajectory_accuracy": 0.0,
            "total_comparisons": 0,
            "correct_trajectories": 0,
            "trajectory_details": []
        }
        
        correct_count = 0
        total_count = 0
        
        for i, (expected, actual) in enumerate(zip(
            outputs["expected_trajectories"], 
            outputs["tool_trajectory"]
        )):
            if i >= len(outputs["questions"]):
                break
                
            total_count += 1
            
            # 标准化轨迹格式进行比较
            expected_clean = expected.strip() if expected else ""
            actual_clean = actual.strip() if actual else ""
            
            # 检查轨迹是否匹配
            is_correct = expected_clean == actual_clean
            
            if is_correct:
                correct_count += 1
            
            trajectory_detail = {
                "question_index": i,
                "question": outputs["questions"][i][:50] + "..." if len(outputs["questions"][i]) > 50 else outputs["questions"][i],
                "expected_trajectory": expected_clean,
                "actual_trajectory": actual_clean,
                "is_correct": is_correct,
                "response_status": "ERROR" if outputs["responses"][i].startswith("ERROR") else "SUCCESS"
            }
            
            trajectory_accuracy_analysis["trajectory_details"].append(trajectory_detail)
        
        if total_count > 0:
            trajectory_accuracy_analysis["trajectory_accuracy"] = correct_count / total_count
            trajectory_accuracy_analysis["total_comparisons"] = total_count
            trajectory_accuracy_analysis["correct_trajectories"] = correct_count
        
        print(f"📊 轨迹准确率: {trajectory_accuracy_analysis['trajectory_accuracy']:.2%} ({correct_count}/{total_count})")
        
        return trajectory_accuracy_analysis
    
    def generate_html_report(self, outputs, ragas_results, trajectory_analysis, trajectory_accuracy_analysis):
        """生成 HTML 报告"""
        print("📄 生成 HTML 报告...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2E 测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .trajectory {{ font-family: monospace; background: #f8f9fa; padding: 5px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 E2E 测试报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 测试概览</h2>
        <p><strong>总问题数:</strong> {trajectory_analysis['total_questions']}</p>
        <p><strong>成功运行:</strong> <span class="success">{trajectory_analysis['successful_runs']}</span></p>
        <p><strong>失败运行:</strong> <span class="error">{trajectory_analysis['total_questions'] - trajectory_analysis['successful_runs']}</span></p>
        <p><strong>成功率:</strong> <span class="success">{(trajectory_analysis['successful_runs'] / trajectory_analysis['total_questions'] * 100):.1f}%</span></p>
    </div>
    
    <div class="section">
        <h2>📈 RAGAS 评估结果</h2>
        <div>
"""
        
        # 添加 RAGAS 指标
        # 只展示均值
        for metric_name, score in ragas_results.items():
            if isinstance(score, list):
                mean_score = float(np.mean(score)) if len(score) > 0 else 0.0
                color_class = "success" if mean_score > 0.7 else "warning" if mean_score > 0.5 else "error"
                html_content += f'<div class="metric"><strong>{metric_name}均值:</strong> <span class="{color_class}">{mean_score:.4f}</span></div>'
            elif score is not None and isinstance(score, (int, float)):
                color_class = "success" if score > 0.7 else "warning" if score > 0.5 else "error"
                html_content += f'<div class="metric"><strong>{metric_name}:</strong> <span class="{color_class}">{score:.4f}</span></div>'
            elif score is not None:
                html_content += f'<div class="metric"><strong>{metric_name}:</strong> <span class="error">{score}</span></div>'
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 轨迹准确率分析</h2>
        <p><strong>总体准确率:</strong> <span class="success">{trajectory_accuracy:.2%}</span></p>
        <p><strong>正确轨迹数:</strong> {correct_trajectories}/{total_comparisons}</p>
        
        <table>
            <tr><th>问题</th><th>预期轨迹</th><th>实际轨迹</th><th>状态</th></tr>
""".format(
            trajectory_accuracy=trajectory_accuracy_analysis["trajectory_accuracy"],
            correct_trajectories=trajectory_accuracy_analysis["correct_trajectories"],
            total_comparisons=trajectory_accuracy_analysis["total_comparisons"]
        )
        
        for detail in trajectory_accuracy_analysis["trajectory_details"]:
            status_class = "success" if detail["is_correct"] else "error"
            status_text = "✅ 正确" if detail["is_correct"] else "❌ 错误"
            html_content += f"""
            <tr>
                <td>{detail['question']}</td>
                <td><span class="trajectory">{detail['expected_trajectory']}</span></td>
                <td><span class="trajectory">{detail['actual_trajectory']}</span></td>
                <td class="{status_class}">{status_text}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>🛤️ 工具使用统计</h2>
        <table>
            <tr><th>工具名称</th><th>使用次数</th></tr>
"""
        
        for tool, count in trajectory_analysis["tool_usage_stats"].items():
            html_content += f"<tr><td>{tool}</td><td>{count}</td></tr>"
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>⏱️ 执行时间统计</h2>
"""
        
        if trajectory_analysis["execution_time_stats"]:
            stats = trajectory_analysis["execution_time_stats"]
            html_content += f"""
        <p><strong>平均执行时间:</strong> {stats['avg_time']:.2f}秒</p>
        <p><strong>最短执行时间:</strong> {stats['min_time']:.2f}秒</p>
        <p><strong>最长执行时间:</strong> {stats['max_time']:.2f}秒</p>
        <p><strong>总执行时间:</strong> {stats['total_time']:.2f}秒</p>
"""
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>📝 详细结果</h2>
        <table>
            <tr><th>问题</th><th>响应</th><th>工具轨迹</th><th>执行时间</th><th>相关度</th></tr>
"""
        # 兼容不同 key 命名（如 answer_relevancy/response_relevancy/ResponseRelevancy），并打印 debug 信息
        relevancy_scores = None
        for key in ["answer_relevancy", "response_relevancy", "ResponseRelevancy"]:
            if key in ragas_results:
                relevancy_scores = ragas_results[key]
                print(f"[DEBUG] 相关度分数 key: {key}, value: {type(relevancy_scores)} -> {relevancy_scores}")
                break
        if relevancy_scores is None:
            print("[DEBUG] 未找到相关度分数 key，ragas_results:", ragas_results)
            relevancy_scores = []
        for i, (question, response, trajectory, exec_time) in enumerate(zip(
            outputs["questions"], 
            outputs["responses"], 
            outputs["tool_trajectory"], 
            outputs["execution_time"]
        )):
            status_class = "success" if not response.startswith("ERROR") else "error"
            # 取相关性分数
            relevancy_str = "-"
            if isinstance(relevancy_scores, list) and i < len(relevancy_scores):
                relevancy = relevancy_scores[i]
                print(f"[DEBUG] 第{i}条相关度原始值: {repr(relevancy)} 类型: {type(relevancy)}")
                try:
                    relevancy_float = float(relevancy)
                    relevancy_str = f"{relevancy_float:.4f}"
                except (TypeError, ValueError):
                    relevancy_str = str(relevancy) if relevancy is not None else "-"
            elif isinstance(relevancy_scores, (int, float)):
                try:
                    relevancy_float = float(relevancy_scores)
                    relevancy_str = f"{relevancy_float:.4f}"
                except (TypeError, ValueError):
                    relevancy_str = str(relevancy_scores)
            else:
                print(f"[DEBUG] 第{i}条相关度无法识别，relevancy_scores: {repr(relevancy_scores)} 类型: {type(relevancy_scores)}")
            # 工具轨迹展示将 [TOOLSEP] 替换为箭头
            trajectory_display = (trajectory or '无工具调用').replace('[TOOLSEP]', ' → ')
            html_content += f"""
            <tr>
                <td>{question[:100]}{'...' if len(question) > 100 else ''}</td>
                <td class="{status_class}">{response[:200]}{'...' if len(response) > 200 else ''}</td>
                <td><span class="trajectory">{trajectory_display}</span></td>
                <td>{exec_time:.2f}s</td>
                <td>{relevancy_str}</td>
            </tr>
"""
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>🔄 轨迹模式分析</h2>
        <table>
            <tr><th>轨迹模式</th><th>出现次数</th></tr>
"""
        
        for pattern, count in trajectory_analysis["trajectory_patterns"].items():
            html_content += f"<tr><td><span class='trajectory'>{pattern}</span></td><td>{count}</td></tr>"
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        # 保存 HTML 报告
        report_path = Path("tests/reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = report_path / f"e2e_test_report_{timestamp}.html"
        
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✅ HTML 报告已生成: {html_file}")
        return str(html_file)
    
    async def run_complete_evaluation(self, csv_path="tests/data/agent_tech_QA_MCP_data.csv"):
        """运行完整的端到端评估"""
        print("🎯 开始完整端到端评估...")
        
        # 1. 加载测试数据
        questions, expected_responses, expected_trajectories = await self.load_test_data(csv_path)
        if not questions:
            return
        
        # 2. 运行端到端测试
        outputs = await self.run_e2e_tests(questions, expected_trajectories,expected_responses)
        
        # 3. RAGAS 评估
        ragas_results = await self.evaluate_with_ragas(outputs)
        
        # 4. 轨迹分析
        trajectory_analysis = self.analyze_trajectory(outputs)
        
        # 5. 轨迹准确率分析
        trajectory_accuracy_analysis = self.analyze_trajectory_accuracy(outputs)
        
        # 6. 生成 HTML 报告
        report_path = self.generate_html_report(outputs, ragas_results, trajectory_analysis, trajectory_accuracy_analysis)
        
        # 7. 保存详细结果
        results = {
            "outputs": outputs,
            "ragas_results": ragas_results,
            "trajectory_analysis": trajectory_analysis,
            "trajectory_accuracy_analysis": trajectory_accuracy_analysis,
            "report_path": report_path
        }
        
        # 保存 JSON 结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = Path("tests/reports") / f"e2e_test_results_{timestamp}.json"
        
        # 转换不可序列化的对象
        serializable_results = {
            "outputs": outputs,
            "ragas_results": ragas_results,
            "trajectory_analysis": trajectory_analysis,
            "trajectory_accuracy_analysis": trajectory_accuracy_analysis,
            "report_path": report_path
        }
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 详细结果已保存: {json_file}")
        print(f"📊 评估完成！查看报告: {report_path}")
        
        return results

# 主执行函数
async def main():
    """主函数"""
    evaluator = E2ETestEvaluator()
    results = await evaluator.run_complete_evaluation()
    return results

# 如果直接运行此脚本
if __name__ == "__main__":
    asyncio.run(main())

