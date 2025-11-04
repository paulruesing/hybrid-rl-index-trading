import multiprocessing
from ctypes import c_char
from datetime import datetime, timedelta
from src.utils.file_management import TxtConfig

class RobustEventManager:
    """ Triggers events and safely waits for triggers while preventing deadlocks through timeouts. """
    def __init__(self):
        self.event = multiprocessing.Event()
        self.lock = multiprocessing.Lock()
        self.trigger_count = multiprocessing.Value('i', 0)

    def set(self):
        with self.lock:
            self.trigger_count.value += 1
            self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self, timeout=None):
        initial_count = self.trigger_count.value

        if timeout is None:
            # Wait indefinitely
            while True:
                if self.event.wait(timeout=1):  # Short timeout for checks
                    with self.lock:
                        if self.trigger_count.value > initial_count:
                            return True
        else:
            # Wait with timeout
            while timeout > 0:
                if self.event.wait(timeout=min(1, timeout)):  # Short timeout for checks
                    with self.lock:
                        if self.trigger_count.value > initial_count:
                            return True
                timeout -= 1
                if timeout <= 0:
                    return False
            return False

    def clear(self):
        with self.lock:
            self.event.clear()
            self.trigger_count.value = 0


class FortifiedSharedMemory:
    """ Defines two shared memory strings with locks and two robust events with proper timeout conditions. """
    def __init__(self, str_len=3000):
        self.str_len = str_len
        self.shared_input_str = multiprocessing.Array(c_char, str_len)
        self.shared_output_str = multiprocessing.Array(c_char, str_len)

        # Dedicated locks for each shared resource
        self.input_lock = multiprocessing.Lock()
        self.output_lock = multiprocessing.Lock()

        # Enhanced event management
        self.chatbot_input_event = RobustEventManager()
        self.response_ready_event = RobustEventManager()

        # Initialize shared strings safely
        with self.input_lock:
            self.shared_input_str.value = b"\x00" * str_len
        with self.output_lock:
            self.shared_output_str.value = b"\x00" * str_len


def check_schedule(function_schedule: dict[callable, list[int]], was_executed: list[bool]) -> (bool, callable, list[bool]):
    """
    Check which scheduled function should be executed based on the current time.

    Parameters
    ----------
    function_schedule : dict of {callable: list of int, list[int], or None}
        A dictionary mapping functions to their schedule parameters. Each schedule is a list
        of four elements specifying [day, weekday, hour, minute]. Each element can be:
            - an int representing the specific time unit,
            - a list of ints for multiple allowed values,
            - or None if that unit should be ignored in the schedule.
        The time units correspond to:
            - day: day of the month (1-31),
            - weekday: day of the week (0=Monday to 6=Sunday),
            - hour: hour of the day (0-23),
            - minute: minute of the hour (0-59).

    was_executed : list of bool
        A list indicating whether each function in `function_schedule` was already executed
        in the current scheduled time slot. This prevents repeated execution within the same slot.

    Returns
    -------
    tuple of (bool, callable or None, list of bool)
        A tuple containing:
            - bool: True if a function is to be executed now, False otherwise.
            - callable or None: The function to execute if any, else None.
            - list of bool: Updated execution status list reflecting which functions were executed.

    Notes
    -----
    - Only the first function matching the current schedule and not previously executed will be returned.
    - The execution flags in `was_executed` are updated to reflect new executions and reset based on time transitions.
    - Please call this function in a loop while reusing its returns as new arguments.
    """
    now = datetime.now()
    func_to_be_executed = None
    for func_ind, (func, schedule) in enumerate(function_schedule.items()):
        day, weekday, hour, minute = schedule

        execute = True
        # check schedule
        if day is not None:
            if isinstance(day, list):
                if now.day not in day or was_executed[func_ind]:
                    execute = False
            elif now.day != day or was_executed[func_ind]:
                execute = False

        if weekday is not None:
            if isinstance(weekday, list):
                if now.weekday() not in weekday or was_executed[func_ind]:
                    execute = False
            elif now.weekday() != weekday or was_executed[func_ind]:
                execute = False

        if hour is not None:
            if isinstance(hour, list):
                if now.hour not in hour or was_executed[func_ind]:
                    execute = False
            elif now.hour != hour or was_executed[func_ind]:
                execute = False

        if minute is not None:
            if isinstance(minute, list):
                if now.minute not in minute or was_executed[func_ind]:
                    execute = False
            elif now.minute != minute or was_executed[func_ind]:
                execute = False

        if execute:
            func_to_be_executed = func
            was_executed[func_ind] = True

        # reset was_executed for the next scheduled time
        reset = True
        # check only the smallest provided timescale, because that is enough reason to reset
        if minute is not None:
            if isinstance(minute, list) and now.minute - 1 not in minute:
                reset = False
            elif not isinstance(minute, list) and now.minute - 1 != minute:
                reset = False
        elif hour is not None:
            if isinstance(hour, list) and now.hour - 1 not in hour:
                reset = False
            elif not isinstance(hour, list) and now.hour - 1 != hour:
                reset = False
        elif weekday is not None:
            if isinstance(weekday, list) and now.weekday() - 1 not in weekday:
                reset = False
            elif not isinstance(weekday, list) and now.weekday() - 1 != weekday:
                reset = False
        elif day is not None:
            if isinstance(day, list) and now.day - 1 not in day:
                reset = False
            elif not isinstance(day, list) and now.day - 1 != day:
                reset = False

        # reset if necessary:
        if reset: was_executed[func_ind] = False

    # return all working memory and func to be executed:
    return (func_to_be_executed is not None), func_to_be_executed, was_executed


def check_schedule_diary(schedule_diary_manager: TxtConfig, function_schedule: dict, verbose: bool = False,
                         callback=print):
    """
    Check the last execution of each function based on the last execution time and the provided schedule.
    Updates:
    - Uses timestamps for comparison instead of component-wise checks.
    - For each function, calculates the next expected execution datetime after the last execution.
    - Determines if a scheduled run was missed based on 'now > next_expected_time' logic.
    """
    temp_dict = {}

    for function, expected_schedule in function_schedule.items():
        function_name = function.__name__

        if function_name in schedule_diary_manager.settings_dict:
            # Get last execution timestamp (should be in [day, weekday, hour, minute])
            last_execution = schedule_diary_manager.get_as_type(function_name, 'float_list')
            last_exec_dt = datetime(
                year=datetime.now().year,  # Assuming current year
                month=datetime.now().month if int(last_execution[0]) < datetime.now().day else datetime.now().month - 1,  # no month defined in diary
                day=int(last_execution[0]),
                hour=int(last_execution[2]),
                minute=int(last_execution[3])
            )

            now = datetime.now()

            # compute next expected timestamp
            def next_expected_time(after_dt, schedule):
                # schedule: [MONATSTAG, WOCHENTAG, STUNDE, MINUTE] -> einzelne Werte oder Listen, mind. Eines von MONATSTAG/WOCHENTAG definiert, beide None bedeutet tägliche Ausführung
                # derive possible next expected dates after today:
                candidates = []

                # first step: derive expected possible candidate days (month_day, week_day, or daily)
                # month-day overwrites weekday (is checked first)
                if schedule[0] is not None:
                    # Monatstage können Einzelwert oder Liste sein
                    days = [schedule[0]] if not isinstance(schedule[0], list) else schedule[0]
                    # Suche zukünftigen Tag im selben/dem nächsten Monat
                    for delta_month in range(2):  # maximal einen Monat voraus prüfen
                        next_month = (after_dt.month + delta_month - 1) % 12 + 1
                        next_year = after_dt.year + ((after_dt.month + delta_month - 1) // 12)
                        for d in days:
                            try:
                                t = datetime(next_year, next_month, d, 0, 0)
                                if t > after_dt:
                                    candidates.append(t)
                            except (ValueError, OverflowError):
                                continue  # skip feasibility errors (z.B. 31.2.)
                elif schedule[1] is not None:
                    # Wochentag(e): Einzelwert oder Liste [0=mo,...,6=so]
                    wdays = [schedule[1]] if not isinstance(schedule[1], list) else schedule[1]
                    for weekday in wdays:
                        for delta_days in range(1, 8):  # maximal eine Woche voraus prüfen
                            t = after_dt + timedelta(days=delta_days)
                            if t.weekday() == weekday:
                                candidates.append(datetime(t.year, t.month, t.day, 0, 0))
                else:
                    # tägliche Ausführung, nimm einfach nächsten Tag
                    t = after_dt + timedelta(days=1)
                    candidates.append(datetime(t.year, t.month, t.day, 0, 0))

                # second step: find possible hours for candidate day
                final_candidates = []
                for cand in candidates:
                    hours = [schedule[2]] if (schedule[2] is not None and not isinstance(schedule[2], list)) else (
                        schedule[2] if schedule[2] else list(range(24)))
                    minutes = [schedule[3]] if (schedule[3] is not None and not isinstance(schedule[3], list)) else (
                        schedule[3] if schedule[3] else [0])

                    for h in hours:
                        for m in minutes:
                            dt = cand.replace(hour=h, minute=m)
                            if dt > after_dt:
                                final_candidates.append(dt)

                if final_candidates:
                    return min(final_candidates)
                else:
                    # Wenn kein Kandidat gefunden, prüfe nächsten Tag wieder...
                    return now + timedelta(days=365 * 5)  # unrealistisch weit in Zukunft, d.h. "keine nächste erwartet"

            next_dt = next_expected_time(last_exec_dt, expected_schedule)
            now = datetime.now()
            missed_exec = now > next_dt

            # status message:
            weekday_dict = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday", 5: "saturday",
                            6: "sunday"}
            if verbose or missed_exec:
                callback(
                    f"Function {function_name} was last executed on the {int(last_execution[0])}. ({weekday_dict[int(last_execution[1])]}) at {last_execution[2]}h{last_execution[3]}min.")

        else:
            missed_exec = True
            if verbose:
                callback(f"No last execution was logged for {function_name}.")

        check_statement = "This is within the schedule." if not missed_exec else f"*Hence, at least one scheduled execution was missed.*"
        if verbose or missed_exec:
            callback(check_statement)

        temp_dict[function] = missed_exec

    return temp_dict


def verify_schedule(function_schedule: dict[callable, list[int]]) -> None:
    """
    Validate function schedules to ensure no consecutive minutes are assigned.

    This function checks the provided schedule mapping of functions to their execution times.
    Consecutive minute values are not allowed for a single function, since a minute is the smallest
    unit of scheduling considered. If any function’s schedule includes consecutive minutes,
    a ValueError is raised.

    Parameters
    ----------
    function_schedule : dict of callable to list of int
        A mapping where keys are functions and values are lists describing their schedules.
        The fourth element of each schedule is expected to be either:
        - a single integer minute, or
        - a list of integers representing scheduled minutes.

    Raises
    ------
    ValueError
        If any function in the schedule has consecutive minutes defined.
    """
    for func, schedule in function_schedule.items():
        _, _ , _, minute = schedule
        if isinstance(minute, list):
            for entry in minute:
                if entry + 1 in minute:
                    raise ValueError(f"Consecutive minutes are not allowed! Please amend the schedule of {func.__name__}.")


def check_request_mapping(request_map: dict[str, callable], input_str: str, last_chatbot_input_str: str) -> (str, callable, callable, str):
    """
    Map a user request string to corresponding functions or responses.

    This function determines the appropriate action or response for a given user input,
    based on the provided request map. It distinguishes between executive requests
    (prefixed with "do"), descriptive requests, repeated queries, and invalid inputs.

    Parameters
    ----------
    request_map : dict of str to callable
        A mapping from input strings to functions that describe or execute actions.
        Descriptive functions (whose output to send) start with "Describe".
        Executive functions (who are to be executed) start with "Do".
    input_str : str
        The current input string provided by the user.
    last_chatbot_input_str : str
        The previous input string handled by the chatbot, used to check for duplicates.

    Returns
    -------
    tuple of (str, callable or None, callable or None, str or None)
        A tuple containing:
        - input_str : str
            The original input string.
        - executive_function : callable or None
            The function to execute if the input maps to an executive request, otherwise None.
        - descriptive_function : callable or None
            The function to describe the request if applicable, otherwise None.
        - output_str : str or None
            The response string to send to the user, or None if none is needed.
    """
    descriptive_function = executive_function = output_str = None  # to be over-written

    # commands:
    if input_str == last_chatbot_input_str:
        output_str = "You wrote the same query as last time. Please write something different first. This helps to prevent redundant executions."
    elif input_str.lower()[:2] == "do" and input_str.lower() in request_map:
        executive_function = request_map[input_str.lower()]
        output_str = "Done!"

    # descriptions:
    elif input_str.lower() in request_map:
        descriptive_function = request_map[input_str.lower()]
    elif "describe " + input_str.lower().strip() in request_map:
        descriptive_function = request_map["describe " + input_str.lower().strip()]
    elif input_str == "":
        output_str = ""  # empty input -> empty response
    else:
        output_str = "*Possible inputs are:*\n\n" + "\n".join(request_map.keys())

    # return all:
    return input_str, executive_function, descriptive_function, output_str