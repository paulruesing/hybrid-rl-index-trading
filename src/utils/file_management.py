from datetime import datetime
import os
import numpy as np

def file_title(title: str, dtype_suffix=".svg", short=False):
    '''
    Creates a file title containing the current time and a data-type suffix.

    Parameters
    ----------
    title: string
            File title to be used
    dtype_suffix: (default is ".svg") string
            Suffix determining the file type.
    Returns
    -------
    file_title: string
            String to be used as the file title.
    '''
    if short:
        return datetime.now().strftime('%Y%m%d') + " " + title + dtype_suffix
    else:
        return datetime.now().strftime('%Y-%m-%d %H_%M_%S') + " " + title + dtype_suffix

def most_recent_file(directory: str, suffix_to_consider: str = ".csv", file_title_keyword: str = None) -> str:
    """ Works only with file-titles starting with YYYY-MM-DD HH_MM_SS (as created by the file_title method above) """
    if "." not in str(directory).split('/')[-1]:
        file_array, date_array = np.array([]), np.array([])
        for file in os.listdir(directory):
            # check for latest csv with ticker in title
            if file.endswith(suffix_to_consider) and file_title_keyword in file:
                din_datestring = file[:10]
                din_timestring = file[11:19].replace('_', ':')
                date = datetime.fromisoformat(din_datestring + ' ' + din_timestring)
                date_array = np.append(date_array, date)
                file_array = np.append(file_array, file)
        return directory / file_array[date_array.argsort()[-1]]
    else:
        raise NotADirectoryError("Provided path is not a directory (i.e. contains dots)!")