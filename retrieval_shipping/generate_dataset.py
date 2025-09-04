from openai import OpenAI as OpenAIClient
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import os
import dotenv
import time
dotenv.load_dotenv()

def create_golden_dataset(
    client: OpenAIClient, 
    model: str,
    documents: List[str], 
    ids: List[str],
    context: str,
    example_queries: str,
    language: str
) -> pd.DataFrame:
    
    if len(ids) != len(documents):
        raise ValueError("Length of ids must match length of documents")
    
    queries: List[str] = []
    # TODO: 系统指令，需要让AI生成测试集
    SYSTEM_INSTRUCTION = f"""
        
        """

    for id, document in tqdm(zip(ids, documents), total=len(ids), desc="Generating queries"):
        # TODO: 结合背景信息，文档，案例query，语言四种信息的prompt模板
        PROMPT = f"""
            """

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": PROMPT}
            ]
        )

        queries.append(completion.choices[0].message.content)

    queries_df = pd.DataFrame({"id": ids, "query": queries})

    return queries_df

def load_documents_from_data_folder(data_folder: str) -> List[str]:
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
                documents.append(file.read())
    return documents

if __name__=="__main__":
    passed_documents = load_documents_from_data_folder("data")

    ids = [i.split("_")[0] for i in os.listdir("data")]

    context = "中远海运(COSCO)新闻与通知的回答"

    example_queries = ["港口关闭后，我的在途货物会如何处理？",
                    "有哪些新的危险品申报要求需要特别注意？",
                    "为什么现在需要申报塑料颗粒信息，这会影响我的运输安排吗？",
                    'COSCO SHIPPING Lines declared "force majeure"; what impact will this have on my cargo?',
                    "Where can I find a complete guide for proper cargo declaration?"
                    ]

    openai_client = OpenAIClient(    
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"))

    golden_dataset_zh = create_golden_dataset(
        client=openai_client,
        model="qwen-flash",
        documents=passed_documents,
        ids=ids,
        context=context,
        example_queries=example_queries,
        language="zh"
    )
    golden_dataset_en = create_golden_dataset(
        client=openai_client,
        model="qwen-flash",
        documents=passed_documents,
        ids=ids,
        context=context,
        example_queries=example_queries,
        language="en"
    )

    pd.concat([golden_dataset_zh, golden_dataset_en]).to_csv("golden_dataset.csv", index=False)