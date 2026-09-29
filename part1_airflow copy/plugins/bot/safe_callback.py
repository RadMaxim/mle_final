from functools import wraps


def safe_callback(func):
    @wraps(func)
    def wrapper(context):
        try:
            return func(context)
        except Exception as e:
            print(f"Callback error in {func.__name__}: {repr(e)}")
            return None

    return wrapper