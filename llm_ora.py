import openai
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain
import cx_Oracle
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

openai.organization = "org-YfHz35c0f0VPg1kimN26kfBl"
openai.api_key = "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"

# It is suitable for inquiring the information of employees of the company, such as personal information of employees, job title, country and region, etc。
# please use SQL query extracting info to answer the user's question.
# SQL should be written using this database schema:
# Table: REGIONS
# Columns: REGION_ID, REGION_NAME
# Table: COUNTRIES
# Columns: COUNTRY_ID, COUNTRY_NAME, REGION_ID
# Table: LOCATIONS
# Columns: LOCATION_ID, STREET_ADDRESS, POSTAL_CODE, CITY, STATE_PROVINCE, COUNTRY_ID
# Table: DEPARTMENTS
# Columns: DEPARTMENT_ID, DEPARTMENT_NAME, MANAGER_ID, LOCATION_ID
# Table: JOBS
# Columns: JOB_ID, JOB_TITLE, MIN_SALARY, MAX_SALARY
# Table: EMPLOYEES
# Columns: EMPLOYEE_ID, FIRST_NAME, LAST_NAME, EMAIL, PHONE_NUMBER, HIRE_DATE, JOB_ID, SALARY, COMMISSION_PCT, MANAGER_ID, DEPARTMENT_ID
# Table: JOB_HISTORY
# Columns: EMPLOYEE_ID, START_DATE, END_DATE, JOB_ID, DEPARTMENT_ID
# The query should be returned in plain text, not in JSON.
# question is : Check the history relationship between each employee and the job

# def get_ora_cursor():
#     center_conn = ''
#     center_cur = ''
#     try:
#         center_conn = cx_Oracle.connect('dmuser', 'dmuser','192.168.30.210:1521' + '/' + 'dmdb')
#         center_cur = center_conn.cursor()
#         return center_cur
#     except Exception as msg:
#         exit
#     finally:
#         center_cur.close()
#         center_conn.close()

class Oracle_llm:
    def __init__(self):
        self.center_conn = cx_Oracle.connect('dmuser', 'dmuser','192.168.30.210:1521' + '/' + 'dmdb')
        self.center_cur = self.center_conn.cursor()
    

    def get_table_names(self):
        """返回一个包含所有表名的列表"""
        table_names = []  # 创建一个空的表名列表
        # 执行SQL查询，获取数据库中所有表的名字
        tables = self.center_cur.execute("select table_name from dba_tables where owner='HR'")
        # 遍历查询结果，并将每个表名添加到列表中s
        for table in tables.fetchall():
            table_names.append(table[0])
        return table_names  # 返回表名列表

    def get_column_names(self, table_name):
        """返回一个给定表的所有列名的列表"""
        column_names = []  # 创建一个空的列名列表
        # 执行SQL查询，获取表的所有列的信息
        #select column_name from dba_tab_columns where table_name='JOBS' and owner='HR'
        columns = self.center_cur.execute("""select column_name from dba_tab_columns where table_name='"""+table_name+"""' and owner='HR'""").fetchall()
        # 遍历查询结果，并将每个列名添加到列表中
        for col in columns:
            column_names.append(col[0].replace(',',''))
        
        return column_names  # 返回列名列表
    
    def get_database_info(self) -> str:
        """返回一个字典列表，每个字典包含一个表的名字和列信息"""
        table_dicts = []  # 创建一个空的字典列表
        # 遍历数据库中的所有表
        for table_name in self.get_table_names():
            columns_names = self.get_column_names(table_name)  # 获取当前表的所有列名
            # 将表名和列名信息作为一个字典添加到列表中
            table_dicts.append({"table_name": table_name, "column_names": columns_names})
        return table_dicts  # 返回字典列表

   


    # hr_template = """
    # It is suitable for inquiring the information of employees of the company, such as personal information of employees, job title, country and region, etc。
    # please use SQL query extracting info to answer the user's question.
    # SQL should be written using this database schema:
    # """+database_schema_string+"""
    # The query should be returned in plain text, not in JSON.{input}"""
    # print(hr_template)

    # prompt_infos = [
    #     {
    #         "name": "员工信息查询",
    #         "description": "适用于公司员工的信息的查询,例如员工个人信息,工作职务,所处的国家和地区等",
    #         "prompt_template": hr_template,
    #     },
    # ]

    # prompt = PromptTemplate(template=hr_template,input_variables=["input"])

    # prompt.format(input="查询每个员工和工作的关系")
    # chain = LLMChain({llm:db_chain, prompt:prompt})

    def ai_generate_sql(self,query):
        database_schema_dict = self.get_database_info()
        database_schema_string = "\n".join(
            [
                f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
                for table in database_schema_dict
            ]
        )
        db = SQLDatabase.from_uri("oracle+cx_oracle://hr:hr@192.168.30.210:1521/?service_name=dmdb")
        llm = OpenAI(temperature=0, verbose=True)
        db_chain = SQLDatabaseChain.from_llm(llm, db,verbose=True,return_sql=True,use_query_checker=True,top_k=0)

        hr_template = """
        It is suitable for inquiring the information of employees of the company, such as personal information of employees, job title, country and region, etc。
        please use SQL query extracting info to answer the user's question.  At the same time, you are an expert in the SQL query language, please check the syntax of the generated SQL query statements, such as removing redundant punctuation marks, and rewrite into a simplified syntax form
        SQL should be written using this database schema:
        """+database_schema_string+"""
        The query should be returned in plain text, not in JSON.
        question is : """+query+"""  ,Please remove the last semicolon"""

        # question is : 请查出与Canada有关系的地区和部门, 要求需要查询出所有国家的信息,即便国家信息在地区或者部门不存在"""请查出与Canada有关系的地区和部门
        # hr_template = """
        # question is : 请查出国家、地区和部门之间的关系"""
        try:
            result = db_chain(hr_template)
            return result['result']
        except Exception as e:
            return e #]#db_chain.run(hr_template)



# pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}?options=-c%20search_path={schema_name}"

# db = SQLDatabase.from_uri(pg_uri)

# gpt = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY,
#                  model_name='gpt-3.5-turbo')

# toolkit = SQLDatabaseToolkit(db=db, llm=gpt)

# agent_executor = create_sql_agent(
#     llm=gpt,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )

# question = "What is the monthly trend of total supplies ?"

# result = agent_executor.run(question)
# print(result)

