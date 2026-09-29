from airflow.providers.telegram.hooks.telegram import TelegramHook # импортируем хук телеграма
from bot.safe_callback import safe_callback
CHAT_ID = "-5144673049"

def check_task(context):
    print("+++"*60)
    ti = context["ti"]

    task_id = ti.task_id
    dag_id = ti.dag_id
    run_id = ti.run_id
    state = ti.state
    message = f"Task {task_id} в DAG {dag_id} завершилась со статусом {state}. run_id={run_id}"
    hook = TelegramHook(token='8665336697:AAEjrD-3iS5qLOnWTeWDQx4a5Fr3z2MRr64', chat_id=CHAT_ID)
    hook.send_message({
        'chat_id': CHAT_ID,
        'text': message
    })
   
    print("+++"*60)
@safe_callback
def send_telegram_success_message(context): # на вход принимаем словарь с контекстными переменными
    ti = context["ti"]
    dag_id = ti.dag_id
    run_id = ti.run_id
    state = ti.state
    message = f"DAG {dag_id} завершился со статусом {state}. run_id={run_id}"
    hook = TelegramHook(token='8665336697:AAEjrD-3iS5qLOnWTeWDQx4a5Fr3z2MRr64', chat_id=CHAT_ID)
    hook.send_message({
        'chat_id': CHAT_ID,
        'text': message
    })
@safe_callback
def send_telegram_failure_message(context):
    ti = context["ti"]
    dag_id = ti.dag_id
    run_id = ti.run_id
    state = ti.state
    message = f"DAG {dag_id} завершился со статусом {state}. run_id={run_id}"
    hook = TelegramHook(token='8665336697:AAEjrD-3iS5qLOnWTeWDQx4a5Fr3z2MRr64', chat_id=CHAT_ID)
    hook.send_message({
        'chat_id': CHAT_ID,
        'text': message
    })