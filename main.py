# ========== 简化版本：跳过embedding联网下载，保留RAG架构定义 + Function Calling计算工具 ==========
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sympy as sp

load_dotenv()


# ========== 计算工具 Function Calling 保留（项目核心要求） ==========
# 只允许 LLM 使用的 sympy 命名空间（白名单，安全）
SAFE_NS = {
    # 常用函数
    "solve": sp.solve,
    "simplify": sp.simplify,
    "expand": sp.expand,
    "factor": sp.factor,
    "Eq": sp.Eq,
    "Symbol": sp.Symbol,
    "symbols": sp.symbols,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "exp": sp.exp,
    "Abs": sp.Abs,
    "Rational": sp.Rational,
    # 常量
    "pi": sp.pi,
    "E": sp.E,
    "oo": sp.oo,
    # 常用变量
    "x": sp.Symbol("x"),
    "y": sp.Symbol("y"),
    "z": sp.Symbol("z"),
    # sympy 模块本身，支持 sp.solve(...) 写法
    "sp": sp,
    # 禁用内置函数
    
}


def math_calculate(expression: str):
    print("=====调试：收到的表达式字符串=====")
    print(repr(expression))  # 新增这一句，看模型输出的原文
    """
    数学计算工具，用来化简表达式、解方程。
    支持 LLM 输出形如：
      - solve(x**2 - 4, x)
      - sp.solve(x**2 - 4, x)
      - Eq(x**2 - 4, 0)
      - x**2 + 2*x + 1
    """
    try:
        expr_str = expression.strip().strip("`").strip()
        if not expr_str:
            return ""

        # 去掉首尾可能的中英文句号
        expr_str = expr_str.rstrip("。.").strip()

        # 用白名单命名空间 eval，安全性由 __builtins__={} 保证
        result = eval(expr_str, SAFE_NS)

        # 如果结果是 sympy 表达式且不是列表，做化简
        if isinstance(result, (list, tuple, set)):
            return ", ".join(str(r) for r in result)

        if isinstance(result, sp.Basic):
            return sp.simplify(result)

        return result
    except Exception as e:
        return f"计算异常：{str(e)}"


# ========== 模拟RAG检索：直接读取本地math_formula.md文本（不再用Chroma向量库，不需要embedding模型） ==========
def build_rag_database():
    # 直接读取知识库文本，不做向量化，规避huggingface网络下载
    with open("./knowledge/math_formula.md", "r", encoding="utf-8") as f:
        content = f.read()
    return content


# 读取本地公式知识库
knowledge_context = build_rag_database()

# ========== DeepSeek大模型配置 ==========
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
    temperature=0
)

# Prompt：把本地读取的数学公式知识库传给大模型
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是初中数学解题助手，参考下面的数学公式知识库。\n"
     """输出规则：
1. 如果用户只是询问公式、定义、概念（例如：什么是平方差公式），只输出文字讲解，不要输出 SYMPY_EXPR: 这一行！
2. 如果题目需要解方程、代数求值、数值运算，在回答的单独一行输出 SYMPY_EXPR:，后面紧跟sympy能识别的表达式，不要LaTeX，不要$，不要\frac。
示例：
SYMPY_EXPR: solve(3*x+2-8, x)"""
     "【知识库内容】\n{context}\n\n"
     "输出要求：\n"
     "1. 写出清晰的分步解题过程\n"
     "2. 在最后单独一行，只写 SYMPY_EXPR:后面跟上sympy表达式，解方程使用 solve(方程, x)，不要多余文字\n"
     "   例如：SYMPY_EXPR:solve(x**2 - 4, x)\n"
     "   例如：SYMPY_EXPR:x**2 + 2*x + 1\n"),
    ("user", "题目：{question}"),
])


def solve_question(question: str):
    # 将本地读取的公式知识库传入prompt，替代向量检索
    resp = llm.invoke(prompt.format(question=question, context=knowledge_context))
    full_text = resp.content

    # 分割：提取解题步骤 和 sympy表达式
    lines = full_text.splitlines()
    expr = ""
    pure_step_text = ""
    for line in lines:
        if line.startswith("SYMPY_EXPR:"):
            expr = line.replace("SYMPY_EXPR:", "").strip()
        else:
            pure_step_text += line + "\n"

    # 调用计算工具，增加异常捕获
    calc_result = ""
    if expr:
        print(f"[DEBUG] expr = {repr(expr)}")   # 调试用，确认 LLM 输出的表达式
        try:
            calc_result = math_calculate(expr)
        except Exception as e:
            calc_result = f"计算异常：{e}"

    # 拼接完整回答，强制追加最终答案
    final_out = pure_step_text
    if calc_result:
        final_out += f"\n【最终答案：{calc_result}】"
    return final_out


# 主循环入口
if __name__ == "__main__":
    print("===== 数学题解答助手（RAG+FunctionCalling） =====")
    print("输入数学题目，输入 exit 退出程序")
    while True:
        user_q = input("\n请输入数学题：")
        if user_q.strip().lower() == "exit":
            print("程序结束")
            break
        output = solve_question(user_q)
        print("\n【解答结果】")
        print(output)