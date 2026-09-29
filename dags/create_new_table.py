
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pendulum
from airflow.decorators import dag, task
from bot.message import send_telegram_failure_message, send_telegram_success_message,check_task
import yaml
PARAMS_PATH = "/opt/airflow/params.yaml"
@dag(
    dag_id='sprint_1_join_table',
    schedule='@once',
    description="Join flats and buildings tables",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)
def prepare_dataset():
    @task(on_success_callback=check_task)
    def create_table():
        from sqlalchemy import Table, MetaData, Column, Integer, Float, Boolean, inspect
      
        hook = PostgresHook('destination_db')
        db_conn = hook.get_sqlalchemy_engine()
        metadata = MetaData()
        with open("/opt/airflow/params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        db_name = params["name_db_first_dag"]
        data_table = Table(
            db_name,
            metadata,
            Column('flat_id', Integer, primary_key=True),
            Column('building_id', Integer),
            Column('floor', Integer),
            Column('kitchen_area', Float),
            Column('living_area', Float),
            Column('rooms', Integer),
            Column('is_apartment', Boolean),
            Column('studio', Boolean),
            Column('total_area', Float),
            Column('price', Float),
            Column('build_year', Integer),
            Column('building_type_int', Integer),
            Column('latitude', Float),
            Column('longitude', Float),
            Column('ceiling_height', Float),
            Column('flats_count', Integer),
            Column('floors_total', Integer),
            Column('has_elevator', Boolean),
            Column('target', Integer)
        )
        
        # 3. Создаем пустую таблицу, если её еще нет
        if not inspect(db_conn).has_table(data_table.name):
            metadata.create_all(db_conn)

    @task(on_success_callback=check_task)
    def extract():
       
        hook = PostgresHook('destination_db')
        engine = hook.get_sqlalchemy_engine()

        sql = """
    SELECT 
        f.id AS flat_id,
        f.building_id,
        f.floor,
        f.kitchen_area,
        f.living_area,
        f.rooms,
        f.is_apartment,
        f.studio,
        f.total_area,
        f.price,
        b.build_year,
        b.building_type_int,
        b.latitude,
        b.longitude,
        b.ceiling_height,
        b.flats_count,
        b.floors_total,
        b.has_elevator
    FROM flats f
    JOIN buildings b ON b.id = f.building_id;
    """

        data = pd.read_sql(sql, engine)
        return data
        

    @task(on_success_callback=check_task)
    def transform(data: pd.DataFrame):
        with open("/opt/airflow/params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        data["target"] = data[params["target"]].astype(int)
        return data

    @task(on_success_callback=check_task)
    def load(data: pd.DataFrame):
        hook = PostgresHook('destination_db')
        engine = hook.get_sqlalchemy_engine()
        with open("/opt/airflow/params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        
        db_name = params["name_db_first_dag"]
        data.to_sql(
            name=db_name,
            con=engine,
            if_exists='replace',
            index=False
        )

    # Вызов задач
    create_table_task = create_table()
    extract_task = extract()
    transform_task = transform(extract_task)
    load_task = load(transform_task)

    create_table_task >> extract_task >> transform_task >> load_task

prepare_dataset()



