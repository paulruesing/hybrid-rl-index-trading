from datetime import datetime
import os
import numpy as np
from pathlib import Path
from typing import Union

def file_title(title: str, dtype_suffix=".svg", short=False):
    '''
    Creates a file title containing the current time and a data-type suffix.

    Parameters
    ----------
    title: string
            File title to be used
    dtype_suffix: (default is ".svg") string
            Suffix determining the file type.
    short: boolean (default is False)
            If True, doesn't add h_min_sec to title (only Y-m-d).
    Returns
    -------
    file_title: string
            String to be used as the file title.
    '''
    if short:
        return datetime.now().strftime('%Y%m%d') + " " + title + dtype_suffix
    else:
        return datetime.now().strftime('%Y-%m-%d %H_%M_%S') + " " + title + dtype_suffix

def most_recent_file(directory: Union[Path, str], suffix_to_consider: str = None, file_title_keywords: [str] = None) -> str:
    """ Works only with file-titles starting with YYYY-MM-DD HH_MM_SS (as created by the file_title method above) """
    if "." not in str(directory).split('/')[-1]:
        file_array, date_array = np.array([]), np.array([])
        for file in os.listdir(directory):
            # check for latest csv with ticker in title
            if suffix_to_consider is not None:
                if not file.endswith(suffix_to_consider): continue

            # check provided keywords
            if file_title_keywords is not None:  # if provided
                if isinstance(file_title_keywords, str): file_title_keywords = [file_title_keywords]  # convert to list if required
                match = True  # bool to only remain true if all keywords found
                for file_title_keyword in file_title_keywords:
                    if file_title_keyword not in file: match = False
                if not match: continue  # view next file

            din_datestring = file[:10]
            din_timestring = file[11:19].replace('_', ':')
            date = datetime.fromisoformat(din_datestring + ' ' + din_timestring)
            date_array = np.append(date_array, date)
            file_array = np.append(file_array, file)
        try:
            return directory / file_array[date_array.argsort()[-1]]
        except IndexError:
            raise ValueError("Provided directory is empty!")
    else:
        raise NotADirectoryError("Provided path is not a directory (i.e. contains dots)!")