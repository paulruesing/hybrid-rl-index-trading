import time
from functools import wraps
from threading import Thread, Event
from selenium.common.exceptions import WebDriverException

def timed_callback_decorator(callback: callable = print, interval_minutes=5):
    """
    Function decorator that times the wrapped function and executes a callback periodically.

    Parameters
    ----------
    callback : Callable
        A callback function to be executed periodically to report status. The callback should accept a single string argument for the message.

    interval_minutes : int, optional
        The time interval in minutes between consecutive executions of the callback function. Defaults to 1 minute.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Variable to stop the callback thread
            stop_event = Event()

            def report_status():
                # Save the start time and initialize timing variables
                start_time_local = time.time()
                last_callback_time = start_time_local
                current_interval = interval_minutes  # Start with the initial interval

                # Periodically calls the callback with increasing intervals
                while not stop_event.is_set():
                    current_time = time.time()
                    # Check if the current interval has elapsed since the last callback
                    if current_time >= last_callback_time + (current_interval * 60):
                        elapsed_time = current_time - start_time_local

                        # Invoke callback with a status message
                        if not stop_event.is_set():
                            callback(
                                f"Function `{func.__name__}` running for {elapsed_time / 60:.2f} minutes... (next check in {current_interval * 2} minutes)")

                        # Update the last callback time
                        last_callback_time = current_time

                        # Increase the interval by the current interval (exponential growth)
                        current_interval += current_interval

                    # Small sleep to prevent busy waiting
                    time.sleep(1)

            # Start the thread that handles periodic callback execution
            status_thread = Thread(target=report_status, daemon=True)
            status_thread.start()

            try:
                # Execute the main function
                result = func(*args, **kwargs)
            finally:
                # Stop the status thread after the function finishes
                stop_event.set()
                status_thread.join()
            return result

        return wrapper

    return decorator


def retry_decorator(exceptions=(ValueError, AttributeError, IndexError, WebDriverException, TypeError, KeyError),
                    on_error_callback: callable = print,
                    retries: int = 2, delay: int = 1):
    """
    Creates a decorator to automatically retry a function upon encountering specified exceptions.

    Parameters
    ----------
    exceptions : tuple, optional
        Exceptions to catch for retrying the function, by default (ValueError, AttributeError, IndexError).
    on_error_callback : callable, optional
        A callback function that receives error messages when an exception occurs, by default None.
    retries : int, optional
        The maximum number of retry attempts before raising the exception, by default 3.
    delay : int, optional
        The delay in seconds between retry attempts, by default 1.

    Returns
    -------
    callable
        A decorator that wraps the specified function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    error_message = f"Starting re-try {attempts}/{retries} because of error: {str(e)}"
                    on_error_callback(error_message)

                    if attempts >= retries:
                        raise  # Re-raise the exception after max retries
                    time.sleep(delay)  # Optional delay between retries

        return wrapper

    return decorator