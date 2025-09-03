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
    
    SYSTEM_INSTRUCTION = f"""
        您是一名专门用于生成查询以整理高质量合成数据集的助理。

        只需输出查询内容，无需任何额外文字或格式。
        """

    for id, document in tqdm(zip(ids, documents), total=len(ids), desc="Generating queries"):
        PROMPT = f"""
            请考虑以下背景信息：
            {context}

            根据以下文本片段：
            <text>
            {document}
            <text>

            请生成一个用户可能会提出的、与上述信息相关的现实查询。

            以下是一些用户提问过的示例查询，您在生成查询时应予以参考：
            <example-queries>
            {example_queries}
            <example-queries>

            请勿重复示例查询，它们仅用于帮助您了解用户提问的类型。请确保您的查询与上述提供的信息相关，并保持与示例查询相似的风格，即不一定总是完整的问句形式。

            只需输出查询内容，无需任何额外文字。你的输出语言必须为{language}。
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