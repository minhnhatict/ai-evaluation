from openai import OpenAI
import pandas as pd

import duckdb
from pydantic import BaseModel, Field
# from IPython.display import Markdown

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend

openai_api_key = "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"
client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-4o-mini"

# define the path to the transactional data
TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_Elasticity_Promotions_Data.parquet'

# prompt template for step 2 of tool 1
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

# code for step 2 of tool 1
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt,
                                                    columns=columns,
                                                    table_name=table_name)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    return response.choices[0].message.content


# code for tool 1
def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL"""
    try:

        # define the table name
        table_name = "sales"

        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        # step 2: generate the SQL code
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        # clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")

        # step 3: execute the SQL query
        result = duckdb.sql(sql_query).df()

        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"

# Construct prompt based on analysis type and data subset
DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

# code for tool 2
def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"

example_data = lookup_sales_data("Show me all the sales for store 1320 on November 1st, 2021")
# print(example_data)

# print(analyze_sales_data(prompt="what trends do you see in this data",
#                          data=example_data))

# prompt template for step 1 of tool 3
CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

# class defining the response format of step 1 of tool 3
class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")

# code for step 1 of tool 3

def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """Ensure datetime columns are not used for summation."""
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data, visualization_goal=visualization_goal)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    try:
        content = response.choices[0].message.content

        # Validate that x_axis is not a datetime column
        if content.x_axis.lower() in ["date", "timestamp", "datetime"]:
            content.x_axis = "product_sku"  # Or another suitable category

        # Ensure y_axis is numeric
        if content.y_axis.lower() in ["date", "timestamp", "datetime"]:
            content.y_axis = "sales"  # Replace with numeric metric

        return {
            "chart_type": content.chart_type,
            "x_axis": content.x_axis,
            "y_axis": content.y_axis,
            "title": content.title,
            "data": data
        }
    except Exception:
        return {
            "chart_type": "bar",
            "x_axis": "product_sku",
            "y_axis": "sales",
            "title": visualization_goal,
            "data": data
        }


# prompt template for step 2 of tool 3
CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""

# code for step 2 of tool 3
def create_chart(config: dict) -> str:
    """Create a chart based on the configuration"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()

    # Ensure the generated code saves the figure instead of showing it
    if "plt.show()" in code:
        code = code.replace("plt.show()", "plt.savefig('lab-1/output_chart.png')")

    return code

# code for tool 3
def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal"""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    return code

code = generate_visualization(example_data,
                              "A bar chart of sales by product SKU. Put the product SKU on the x-axis and the sales on the y-axis.")
print(code)
exec(code)